import torch
import torch.nn.functional as F
from pytorch3d.renderer import (
    PerspectiveCameras,
    MeshRasterizer,
    MeshRenderer,
    SoftSilhouetteShader,
    RasterizationSettings,
    BlendParams,
    HardPhongShader,
    AmbientLights,
)
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import TexturesVertex


class PoseRefiner:
    def __init__(self, mesh, K, image_size, device="cuda:0"):
        self.device = torch.device(device)
        self.mesh = mesh.to(self.device)
        self.mesh.scale_verts_(0.001)  # scale mm -> meters
        self.h, self.w = image_size

        # Camera intrinsics
        f_x, f_y = K[0, 0], K[1, 1]
        p_x, p_y = K[0, 2], K[1, 2]
        self.focal = torch.tensor(
            (f_x, f_y), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        self.principal = torch.tensor(
            (p_x, p_y), dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        self.image_size = torch.tensor([[self.h, self.w]], device=self.device)

        self.raster_settings = RasterizationSettings(
            image_size=(self.h, self.w),
            blur_radius=0.0,
            faces_per_pixel=1,
            perspective_correct=True,
            max_faces_per_bin=self.mesh.faces_packed().shape[0],
        )

        self.lights = AmbientLights(device=self.device)

        self.blend_params = BlendParams(
            sigma=1e-4, gamma=1e-4, background_color=(0.0, 0.0, 0.0)
        )

        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=self.raster_settings),
            shader=HardPhongShader(device=self.device, lights=self.lights),
        )

    def render_depth(self, R, T):
        RT = torch.zeros((4, 4))
        RT[3, 3] = 1
        RT[:3, :3] = R
        RT[:3, 3] = T

        Rz = torch.tensor(
            [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        ).float()

        RT = torch.matmul(Rz, RT)

        R = RT[:3, :3].t().reshape(1, 3, 3)
        T = RT[:3, 3].reshape(1, 3)

        cameras = PerspectiveCameras(
            R=R,
            T=T,
            focal_length=self.focal,
            principal_point=self.principal,
            image_size=self.image_size,
            device=self.device,
            in_ndc=False,
        )

        fragments = MeshRasterizer(
            cameras=cameras, raster_settings=self.raster_settings
        )(self.mesh)
        depth = fragments.zbuf[..., 0]  # shape: (1, H, W)

        return depth

    def refine_pose(
        self,
        init_R,
        init_T,
        gt_depth,
        num_iters=300,
        lr=1e-5,
        visualize_every=1,
        min_delta=1e-5,
        patience=30,
    ):
        # Converti le pose iniziali in parametri ottimizzabili
        # Usiamo parametri angolo-assale (rodrigues) per la rotazione per evitare problemi di ortogonalità
        init_rot = self.matrix_to_rodrigues(init_R)
        rot_params = init_rot.clone().detach().to(self.device).requires_grad_(True)
        trans_params = init_T.clone().detach().to(self.device).requires_grad_(True)

        optimizer = torch.optim.Adam([rot_params, trans_params], lr=lr)

        loss_history = []
        best_rot_params = None
        best_T = None
        best_loss = float("inf")
        patience_counter = 0

        for i in range(num_iters):
            optimizer.zero_grad()

            # Converti i parametri rodrigues in matrice di rotazione
            R = self.rodrigues_to_matrix(rot_params)
            T = trans_params

            pred_depth = self.render_depth(R.unsqueeze(0), T.unsqueeze(0))

            valid_mask = (gt_depth > 0) & (pred_depth > 0)
            loss = F.mse_loss(pred_depth[valid_mask], gt_depth[valid_mask])
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

            if visualize_every and i % visualize_every == 0:
                print(f"[{i}/{num_iters}] Loss: {loss.item():.6f}")

            current_loss = loss.item()
            if current_loss + min_delta < best_loss:
                best_loss = current_loss
                patience_counter = 0
                best_rot_params = rot_params.detach().clone()
                best_T = T.detach().clone()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(
                        f"Early stopping at iteration {i}, best loss: {best_loss:.6f}"
                    )
                    break
        # Final pose
        with torch.no_grad():
            R_final = self.rodrigues_to_matrix(best_rot_params).unsqueeze(0)
            T_final = best_T.unsqueeze(0)

        return R_final, T_final, loss_history

    def matrix_to_rodrigues(self, R):
        """Convert rotation matrix to Rodrigues vector"""
        theta = torch.acos((torch.trace(R) - 1) / 2)
        if theta < 1e-6:
            return torch.zeros(3, device=R.device)
        else:
            K = (R - R.T) / (2 * torch.sin(theta))
            return theta * torch.tensor([K[2, 1], K[0, 2], K[1, 0]], device=R.device)

    def rodrigues_to_matrix(self, r):
        """Convert Rodrigues vector to rotation matrix"""
        theta = torch.norm(r)
        if theta < 1e-6:
            return torch.eye(3, device=r.device)
        else:
            k = r / theta
            K = torch.tensor(
                [[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], device=r.device
            )
            return (
                torch.eye(3, device=r.device)
                + torch.sin(theta) * K
                + (1 - torch.cos(theta)) * (K @ K)
            )

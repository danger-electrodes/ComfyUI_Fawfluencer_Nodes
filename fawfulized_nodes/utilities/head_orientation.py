import torch
import numpy as np
import cv2
import mediapipe as mp

def get_closest_face_by_orientation(image, reference_images):
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, min_detection_confidence=0.5, max_num_faces=1)
        input_orientations = analyze_orientations(face_mesh, image)
        reference_orientations = analyze_orientations(face_mesh, reference_images)

        sorted_images = get_best_face_orientation_match(reference_images, input_orientations, reference_orientations)

        return sorted_images

def analyze_orientations(face_mesh, images):
    orientations = []
    for idx in range(images.shape[0]):
        img = images[idx].cpu().numpy()  # shape: HWC (after slicing batch)

        # Normalize if necessary
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)

        # Make sure it's only 3 channels (drop alpha or extras)
        if img.shape[2] > 3:
            img = img[:, :, :3]

        # Convert to RGB (if needed — depends on what color format original was)
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            face_3d = []
            face_2d = []
            for lmk_idx, lm in enumerate(face_landmarks.landmark):
                if lmk_idx in [33, 263, 1, 61, 291, 199]:
                    x, y = int(lm.x * img.shape[1]), int(lm.y * img.shape[0])
                    face_3d.append([x, y, lm.z])
                    face_2d.append([x, y])

            face_3d = np.array(face_3d, dtype=np.float64)
            face_2d = np.array(face_2d, dtype=np.float64)

            focal_length = 1 * img.shape[1]
            cam_matrix = np.array([[focal_length, 0, img.shape[1] / 2],
                                   [0, focal_length, img.shape[0] / 2],
                                   [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            success, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            rmat, _ = cv2.Rodrigues(rot_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            x, y, z = angles[0] * 360, angles[1] * 360, angles[2] * 360
            orientations.append((x, y, z))
        else:
            orientations.append((0, 0, 0))

    return orientations

def get_best_face_orientation_match(reference_images, input_orientations, reference_orientations):
    # Since input_orientations always has length 1, we directly take the first orientation.
    input_orientation = input_orientations[0]

    best_diff = float('inf')
    best_index = -1

    # Iterate over reference orientations to find the closest match
    for ref_idx, ref_orientation in enumerate(reference_orientations):
        # Compute the Euclidean distance between input_orientation and ref_orientation
        diff = np.linalg.norm(np.subtract(input_orientation, ref_orientation))
        
        # Update best match if current diff is smaller than the previous best
        if diff < best_diff:
            best_diff = diff
            best_index = ref_idx

    # Get the best matching image from reference_images
    closest_face_orientation_image = reference_images[best_index]

    # Ensure the output shape is [1, H, W, C] by unsqueezing the batch dimension
    closest_face_orientation_image = torch.unsqueeze(closest_face_orientation_image, 0)

    return closest_face_orientation_image

def format_orientation_data(orientations):
    data_output = ""
    for x, y, z in orientations:
        data_output += f"[{x:.2f},{y:.2f},{z:.2f}]\n"
    return data_output
from app.services.posture_analysis import extract_landmarks, calculate_angle

def extract_angles(landmarks):
    try:
        knee = calculate_angle(landmarks[23], landmarks[25], landmarks[27])
        spine = calculate_angle(landmarks[11], landmarks[23], landmarks[25])
        elbow = calculate_angle(landmarks[11], landmarks[13], landmarks[15])

        return {
            "knee": knee,
            "spine": spine,
            "elbow": elbow
        }
    except:
        return None


def build_templates(dataset):
    templates = {}

    for pose_name, images in dataset.items():

        angle_list = []

        for img in images:
            landmarks = extract_landmarks(img)

            if landmarks is None:
                continue

            angles = extract_angles(landmarks)

            if angles:
                angle_list.append(angles)

        if len(angle_list) == 0:
            continue

        # Average angles
        avg_knee = sum(a["knee"] for a in angle_list) / len(angle_list)
        avg_spine = sum(a["spine"] for a in angle_list) / len(angle_list)
        avg_elbow = sum(a["elbow"] for a in angle_list) / len(angle_list)

        templates[pose_name] = {
            "knee": round(avg_knee, 2),
            "spine": round(avg_spine, 2),
            "elbow": round(avg_elbow, 2)
        }

    return templates
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)

def extract_hand_landmarks(image):
    """Extract hand landmarks from an image.
    
    Args:
        image: OpenCV image (BGR format)
        
    Returns:
        List of hand landmark lists. Each hand has 21 landmarks with x, y, z coordinates.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    hands_landmarks = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            hand_data = []
            for lm in hand_landmarks.landmark:
                hand_data.append({
                    "x": float(lm.x),
                    "y": float(lm.y),
                    "z": float(lm.z)
                })
            hands_landmarks.append(hand_data)

    return hands_landmarks
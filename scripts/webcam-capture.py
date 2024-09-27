import cv2

def capture_and_detect():
    # Charger le classificateur en cascade pré-entraîné pour la détection de visage
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Initialiser la webcam
    cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("Impossible d'ouvrir la webcam")
        return
    
    while True:
        # Capturer une image
        ret, frame = cap.read()
        
        if not ret:
            print("Échec de la capture d'image")
            break
        
        # Convertir l'image en niveaux de gris
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Détecter les visages
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Dessiner un rectangle autour de chaque visage détecté
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Afficher l'image
        cv2.imshow('Face Detection', frame)
        
        # Attendre la touche 'q' pour quitter
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Libérer la webcam et fermer les fenêtres
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_detect()
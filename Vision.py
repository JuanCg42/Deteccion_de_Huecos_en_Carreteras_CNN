import cv2
import numpy as np
import tensorflow as tf
                                                                                                                                                                                                                
modelo = tf.keras.models.load_model("modelo_huecos.keras") 


clases = ["Grieta", "Normal", "Bache", "Erosion superficial"]

camara = cv2.VideoCapture(1) 
contador = 0                 

while True:
    ok, frame = camara.read() 
    if not ok:
        print("No se pudo leer la camara")
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR) 
    img = img.astype("float32") / 255            

    pred = modelo(img.reshape(1, 224, 224, 3), training=False).numpy()[0] 
    idx = np.argmax(pred)

    texto = f"{clases[idx]}: {pred[idx]*100:.1f}%"
    color = (0, 255, 0) if clases[idx] == "Normal" else (0, 0, 255) 
    cv2.putText(frame, texto, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2) 

    cv2.imshow("Deteccion de huecos", frame) 

    tecla = cv2.waitKey(1) & 0xFF
    if tecla == ord("s"): 
        contador += 1
        cv2.imwrite(f"captura_{contador}.png", frame)
        print("Captura guardada:", f"captura_{contador}.png")
    if tecla == ord("q"): 
        break

camara.release()
cv2.destroyAllWindows()
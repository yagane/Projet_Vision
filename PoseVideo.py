import cv2 as cv

net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

inWidth = 368
inHeight = 368
prob = 0.2

BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                   "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                   "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Background": 14}

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"]]

video = cv.VideoCapture('2022-12-05_17-04-33.mp4')

video.set(cv.CAP_PROP_FPS, 20)

while cv.waitKey(1) < 0:
    hasFrame, frame = video.read()
    if not hasFrame:
        cv.waitKey()
        break

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :15, :, :]

    points = []
    for i in range(len(BODY_PARTS)):
        # On récupère la heat map de la partie du corps voulu
        heatMap = out[0, i, :, :]

        # On récupère un maximum global
        _, conf, _, point = cv.minMaxLoc(heatMap)
        # On replace le point sur l'image
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]

        # Ajoute le point si la probabilité qu'il soit bon est supérieur à thr
        points.append((int(x), int(y)) if conf > prob else None)

    for pair in POSE_PAIRS:
        # On récupère les 2 parties du corps voulu
        part1 = pair[0]
        part2 = pair[1]
        num1 = BODY_PARTS[part1]
        num2 = BODY_PARTS[part2]

        # Si les deux parties sont dans points on dessine les points et on les relies à l'aide d'une ligne
        if points[num1] and points[num2]:
            cv.line(frame, points[num1], points[num2], (0, 255, 0), 3)
            cv.ellipse(frame, points[num1], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[num2], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

    cv.imshow('Pose estimation tutoriel', frame)
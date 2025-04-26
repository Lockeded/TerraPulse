import numpy as np
import cv2 as cv


def visualize(image, faces):
    output = image.copy()
    for idx, face in enumerate(faces):
        coords = face[:-1].astype(np.int32)
        # Draw face bounding box
        cv.rectangle(output, (coords[0], coords[1]), (coords[0] + coords[2], coords[1] + coords[3]), (0, 255, 0), 2)
        # Draw landmarks
        cv.circle(output, (coords[4], coords[5]), 2, (255, 0, 0), 2)
        cv.circle(output, (coords[6], coords[7]), 2, (0, 0, 255), 2)
        cv.circle(output, (coords[8], coords[9]), 2, (0, 255, 0), 2)
        cv.circle(output, (coords[10], coords[11]), 2, (255, 0, 255), 2)
        cv.circle(output, (coords[12], coords[13]), 2, (0, 255, 255), 2)
        # Put score
        cv.putText(output, '{:.4f}'.format(face[-1]), (coords[0], coords[1] + 15), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                   (0, 255, 0))

    return output


def main(model, image):
    score_threshold = 0.85
    nms_threshold = 0.3
    backend = cv.dnn.DNN_BACKEND_DEFAULT
    target = cv.dnn.DNN_TARGET_CPU

    # Instantiate yunet
    yunet = cv.FaceDetectorYN.create(
        model=model,
        config='',
        input_size=(320, 320),
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
        top_k=5000,
        backend_id=backend,
        target_id=target
    )

    yunet.setInputSize((image.shape[1], image.shape[0]))
    _, faces = yunet.detect(image)  # faces: None, or nx15 np.array
    vis_image = visualize(image, faces)

    vis = True
    if vis:
        # cv.namedWindow('xx', cv.WINDOW_AUTOSIZE)
        cv.imshow('xx', vis_image)
        cv.waitKey(0)


if __name__ == '__main__':
    model = 'C:/Users/10139/Documents/SelfLearning/ImageConan/TerraPulse/models/base_M/face_detection_yunet_2022mar.onnx'
    image = cv.imread('TerraPulse/people.jpg')
    main(model, image)



# bgslr-video: Bulgarian Sign Language recognition from video (prototype)

Version 2 of [bgslr-static](https://github.com/Sp630/bgslr-static). v1 recognizes the dactylic (fingerspelling) alphabet from single frames. After taking a Bulgarian Sign Language course and talking with Deaf signers about v1 I realized that wasn't enough. Many signs are defined by motion, which a single image can't capture. This version classifies signs from short video sequences instead.

**Status:** prototype, trained on 3 signs: Здравей (hello), Казвам се (my name is), Работя (I work).


## How it works

1. **Landmarks.** MediaPipe's pose landmarker (33 body points) and hand landmarker (21 points per hand, up to two hands) run on the live webcam stream.
2. **Normalization.** Coordinates are expressed relative to the shoulders and scaled by shoulder width, so a sign looks the same to the model regardless of camera distance.
3. **Sequences.** Each frame becomes a 225-value vector (99 pose + 63 per hand, with fixed left/right slots), sampled at about 15 fps into 60-frame windows (~4 seconds). When a detector misses a frame, the previous values carry forward.
4. **Augmentation.** Every recording is also mirrored horizontally, doubling the dataset.
5. **Model.** A 2-layer LSTM (hidden size 128) in PyTorch; its final hidden state goes to a linear classifier.
6. **Live prediction.** A sliding 60-frame window is classified continuously, and the predicted sign is output on screen.

## Files

- `Scripts/DataCollection.py`: records one labeled 60-frame sample from the webcam
- `Scripts/DataCollectionUpdated.py`: rework with time-based sampling (in progress)
- `Scripts/FlipLandmark.py`: mirroring augmentation
- `Scripts/LoadDataset.py`, `Scripts/Training.py`: data loading, model, training loop
- `Scripts/Predict.py`, `Scripts/UI.py`: live recognition with on-screen overlay
- `Models/`: MediaPipe's pretrained hand and pose landmarker models (from Google)

The recorded dataset and trained weights aren't in the repo yet, so running live prediction requires collecting data and training first.

**Requirements:** Python 3, mediapipe, opencv-python, torch, numpy, scikit-learn, Pillow

## Next

- Finish time-based sampling so sequence length doesn't depend on camera frame rate
- Compare the LSTM against a GRU, 1D temporal CNN and a small transformer
- Expand the vocabulary beyond 3 signs and record from more signers

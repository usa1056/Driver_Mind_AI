# Driver Drowsiness Detection

本專案使用 OpenCV + Dlib 進行眼睛閉合與打哈欠疲勞判斷，並透過 Gradio 提供簡易前端互動界面。

## 功能
- 支援 webcam 與影片上傳
- 疲勞警示 + 打哈欠偵測
- 實時顯示 EAR/MAR 指標

## 模型檔案
請先下載 `shape_predictor_68_face_landmarks.dat` 並放在主目錄下（Dlib 官方下載：[連結](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)）。

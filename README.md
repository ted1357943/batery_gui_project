## 使用說明：
先把整包clone 至自己資料夾中,確認是否在ros環境，可參考 [ros30天](https://ithelp.ithome.com.tw/users/20112348/ironman/1965)

---

###  1. 確認conda環境符合QT要求(第一次clone下來要確認)

#### 若還沒有conda環境

執行
```bash
conda env create -f src/battery_detect_withGUI/src/environment.yml
```
#### 若要在現有conda環境下執行

確認現有環境與src/battery_detect_withGUI/src/environment.yml內容，比對版本是否符合需求


---

###  2. 確認conda環境符合yolov5_obb要求(第一次clone下來要確認)

執行
```bash
pip install -r src/battery_detect_withGUI/src/yolov5_obb/requirements.txt
python src/battery_detect_withGUI/src/yolov5_obb/utils/nms_rotatedsetup.py develop
```
可參考 [yolov5_obb/install.md](https://github.com/hukaixuan19970627/yolov5_obb/blob/master/docs/install.md)
(==若報錯，可能是cuda版本不匹配的問題==)

---

###  3. 確認環境安裝上都沒問題後在workspace編譯這包程式

執行
```bash
catkin_make
```
---

### 4. 下面程式裡的路徑要修改為自己存放檔案的路徑

1. yolov5_obb/detect.py
2. Controller/main.py

像是detect.py的
```bash
 weights='/home/yang/jojo/battery_project/battery_detect_ros/src/battery_detect_withGUI/src/yolov5_obb/runs/train/shape0305/weights/best.pt'
```
就要把detect.py的/home/yang/jojo/battery_project/battery_detect_ros/這部分改成自己存放檔案的位置

---
###  5. 都沒任何問題後可直接執行qt程式

在workspace執行
```bash
source devel/setup.bash
roslaunch battery_detect_withGUI battery_process.launch
```


執行roslaunch battery_detect_withGUI battery_process.launch會同時執行下三個指令，也可開3個terminal分別執行上面這3行指令，會跟roslaunch有一樣效果

```bash
roscore
rosrun battery_detect_withGUI src/Controller/main.py
rosrun tm_driver 169.254.213.199
```
 
---
## QT功能說明
#### 1. 相機調整：
此功能用於確認相機拍攝範圍是符合預期，若不符合則手動調整相機位置使畫面恰好拍攝到輸送帶中央。(==需使用realsense相機==)
#### 2. 座標校正：
若有移動到相機就必須進行座標校正，會先跳出一個手臂校正的提示，請先在輸送帶上放上QR code，之後用手臂對到QR code的原點,X點,Y點定出新座標系，確保不會再移動鏡頭與QR code後按下手臂校正提示的OK，按下OK後程式會自動抓取QR code三點當作新座標軸，並在10秒後自動結束校正功能。(==需使用realsense相機==)
#### 3. 參數調整：
調整程式參數。(==需使用realsense相機==)
#### 4. 夾取電池：
執行夾取專案。
#### 5. 停止夾取：
停止程式，手臂回到HOME位置，並於畫面右上方顯示各類電池統計數據。
#### 6. 緊急停止：
停止程式，手臂停在當前位置。
from nuscenes.nuscenes import NuScenes

# nuScenesのインスタンスを作成（バージョンとデータルートを指定）
nusc = NuScenes(version='v1.0-trainval', dataroot='/path/to/nuscenes', verbose=True)

# 処理するサンプルの数を設定
num_samples_to_process = 10  # 例えば10個のサンプルを処理する

# 最初のシーンからサンプルを取得
my_scene = nusc.scene[0]
sample_token = my_scene['first_sample_token']

for _ in range(num_samples_to_process):
    sample = nusc.get('sample', sample_token)

    # カメラ画像の取得
    for cam in ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']:
        cam_data = nusc.get('sample_data', sample['data'][cam])
        cam_filepath = nusc.get_sample_data_path(cam_data['token'])
        # 画像を表示または処理

    # 物体の情報とキャリブレーションデータの取得
    for ann_token in sample['anns']:
        ann_data = nusc.get('sample_annotation', ann_token)
        center = ann_data['translation']
        size = ann_data['size']
        # 中心座標とサイズの処理

        calib_data = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        # キャリブレーションデータの処理

    # 次のサンプルに進む
    if sample['next']:
        sample_token = sample['next']
    else:
        break

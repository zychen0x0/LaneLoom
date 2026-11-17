"""Common training split for Q0 architecture comparison.

Target scenes and every neighboring frame in their segments are excluded.
Train both baseline and spatial prior with this split and ordinary sampling.
No target-specific training, cached epoch-43 features, or OpenLane warm start.
"""
_base_ = ['./laneloom_openlane_image_lidar_temporal_0923.py']

work_dir = './work_dirs/laneloom_openlane_q0_holdout_baseline'
load_from = None
resume = False

q0_target_frames = {'uphill': ['155352583852058800',
            '154999858285296600',
            '155330855180965900',
            '155008311058318500',
            '155361178125380500',
            '154446518417311900'],
 'uphill_unmarked': ['155369952933955400', '155369966522612500', '154950132079864100'],
 'downhill': ['152156189300867700',
              '155430559481466200',
              '155353132144891000',
              '155389914544669900'],
 'curve': ['155352613550456900',
           '150698964368626700',
           '150897652169708900',
           '155266934958181700',
           '155413271986307400',
           '155389913196133000',
           '155389913664942600',
           '151510031514176300']}
q0_excluded_segments = ['segment-1172406780360799916_1660_000_1680_000_with_camera_labels',
 'segment-1208303279778032257_1360_000_1380_000_with_camera_labels',
 'segment-18141076662151909970_2755_710_2775_710_with_camera_labels',
 'segment-18380281348728758158_4820_000_4840_000_with_camera_labels',
 'segment-1863454917318776530_1040_000_1060_000_with_camera_labels',
 'segment-1887497421568128425_94_000_114_000_with_camera_labels',
 'segment-2114574223307001959_1163_280_1183_280_with_camera_labels',
 'segment-2336233899565126347_1180_000_1200_000_with_camera_labels',
 'segment-2935377810101940676_300_000_320_000_with_camera_labels',
 'segment-2961247865039433386_920_000_940_000_with_camera_labels',
 'segment-3224923476345749285_4480_000_4500_000_with_camera_labels',
 'segment-3418007171190630157_3585_530_3605_530_with_camera_labels',
 'segment-3617043125954612277_240_000_260_000_with_camera_labels',
 'segment-4266984864799709257_720_000_740_000_with_camera_labels',
 'segment-5200186706748209867_80_000_100_000_with_camera_labels',
 'segment-6193696614129429757_2420_000_2440_000_with_camera_labels',
 'segment-6694593639447385226_1040_000_1060_000_with_camera_labels',
 'segment-7373597180370847864_6020_000_6040_000_with_camera_labels',
 'segment-8345535260120974350_1980_000_2000_000_with_camera_labels']

train_dataset_cfg = dict(excluded_segments=q0_excluded_segments, scene_cache_size=16)
train_dataloader = dict(dataset=dict(dataset_cfg=train_dataset_cfg))

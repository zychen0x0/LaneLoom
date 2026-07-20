# LaneLoom data layout

LaneLoom owns its dataset location. Place the prepared OpenLane labels,
camera images, and Waymo LiDAR metadata under this directory:

```text
data/
├── openlane/
│   ├── images/
│   └── lane3d_1000/
└── waymo/
    └── waymo_processed_data/
```

For a different storage location, set `LANELOOM_DATA_ROOT` to that directory
before launching a command. In the current container, the configuration also
uses `/workspace/lanedet3d/data` when that mounted data directory exists. It is
treated strictly as data storage; LaneLoom imports no source code from it.

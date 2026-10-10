import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from src.zones import Zone, assign_zone
from src.metrics import compute_metrics
from src.flagging import bottleneck_flags, compare_zone_baseline
from src.calibration import fit_homography, acceptance_gate

def test_homography_and_heldout_error():
    src=np.array([[0,0],[100,0],[100,100],[0,100],[50,0],[100,50],[50,100],[0,50]],float)
    dst=src/100
    H,err=fit_homography(src,dst,[0,1,2,3,4,5],[6,7])
    assert err < 1.0
    assert acceptance_gate(err,30)[0]

def test_zone_buffer_uncertain():
    zones=[Zone("a","browse",Polygon([(0,0),(2,0),(2,2),(0,2)]))]
    assert assign_zone(0.05,1,zones,buffer_m=.15)=="uncertain"
    assert assign_zone(1,1,zones,buffer_m=.15)=="a"

def test_metrics_separate_presence_and_stop():
    df=pd.DataFrame([
      {"timestamp":i*.5,"track_id":1,"x_m":1.,"y_m":1.,"zone":"a","speed_mps":.05,
       "low_quality":False,"in_trust_mask":True} for i in range(20)
    ])
    m=compute_metrics(df,{"a":4},observation_seconds=10)
    row=m["zone_metrics"][0]
    assert row["presence_seconds"] > 0
    assert row["stationary_stop_seconds"] > 0

def test_bottleneck_requires_persistence():
    df=pd.DataFrame([{"zone":"q","density_people_per_m2":2,"median_speed_mps":.1,"low_quality_fraction":0.05}])
    assert bottleneck_flags(df,observed_persistence={"q":30}) == []
    assert len(bottleneck_flags(df,observed_persistence={"q":200})) == 1

def test_dead_zone_insufficient_baseline():
    current=pd.DataFrame([{"zone":"a","track_count_approx":0}])
    baseline=pd.DataFrame([{"zone":"a","track_count_approx":2}])
    assert compare_zone_baseline(current,baseline)[0]["status"]=="insufficient_data"


from src.calibration import is_in_trust_mask


def test_trust_mask_lookup():
    mask = {
        "x_m": [0.0, 1.0, 2.0],
        "y_m": [0.0, 1.0],
        "trusted": [
            [True, False, True],
            [False, True, True],
        ],
    }

    assert is_in_trust_mask(0.0, 0.0, mask) is True
    assert is_in_trust_mask(1.0, 0.0, mask) is False
    assert is_in_trust_mask(1.0, 1.0, mask) is True
    assert is_in_trust_mask(5.0, 0.0, mask) is False
    assert is_in_trust_mask(0.0, 0.0, None) is False
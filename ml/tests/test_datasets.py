from kuulo_ml.datasets import HARD_NEGATIVES, dronenoise_manifest, esc50_manifest, flight_of

STEMS = [
    f"Ed_{drone}_10_F15_N_W_dw_ev{ev}_M{mic}"
    for drone in ("3p", "Fp", "M3", "Yn") for ev in (1, 2, 3, 4) for mic in (1, 2, 3)
] + ["Ed_M3_10_H00_Y_C_nw_ev1_M1", "Ed_M3_10_H00_Y_C_nw_ev1_M2"]


def fake_dronenoise(root):
    folder = root / "dronenoise"
    folder.mkdir(parents=True)
    for stem in STEMS:
        (folder / f"{stem}.wav").touch()
    (folder / "Calib_Scotland_ch1.wav").touch()
    return root


def test_flight_groups_all_microphones_of_one_flight():
    assert flight_of("Ed_M3_10_F15_N_E_dw_ev4_M7") == "Ed_M3_10_F15_N_E_dw_ev4"


def test_dronenoise_manifest_skips_calibration_and_labels_drones(tmp_path):
    clips = dronenoise_manifest(fake_dronenoise(tmp_path))
    assert len(clips) == len(STEMS)
    assert all(c.label == 1 and c.kind.startswith("drone:") for c in clips)
    assert {c.kind for c in clips} == {"drone:3p", "drone:Fp", "drone:M3", "drone:Yn"}


def test_no_flight_is_split_across_train_val_test(tmp_path):
    clips = dronenoise_manifest(fake_dronenoise(tmp_path))
    splits_of: dict[str, set[str]] = {}
    for c in clips:
        splits_of.setdefault(c.group, set()).add(c.split)
    assert all(len(s) == 1 for s in splits_of.values())


def test_every_drone_type_has_test_data(tmp_path):
    clips = dronenoise_manifest(fake_dronenoise(tmp_path))
    for kind in {c.kind for c in clips}:
        assert {c.split for c in clips if c.kind == kind} == {"train", "val", "test"}


def test_esc50_folds_become_splits_and_hard_negatives_are_marked(tmp_path):
    meta = tmp_path / "esc50" / "ESC-50-master" / "meta"
    meta.mkdir(parents=True)
    rows = ["filename,fold,target,category,esc10,src_file,take"]
    for fold in range(1, 6):
        rows.append(f"{fold}-1-A-41.wav,{fold},41,chainsaw,False,{fold}00,A")
        rows.append(f"{fold}-2-A-0.wav,{fold},0,dog,True,{fold}01,A")
    (meta / "esc50.csv").write_text("\n".join(rows))
    clips = esc50_manifest(tmp_path)
    assert {c.split for c in clips if c.path.name.startswith("5-")} == {"test"}
    assert {c.split for c in clips if c.path.name.startswith("4-")} == {"val"}
    assert all(c.label == 0 for c in clips)
    assert "chainsaw" in HARD_NEGATIVES and "dog" not in HARD_NEGATIVES

from engine.campaign_orchestrator import JobStatus, create_campaign
from state import campaign_storage
from io import BytesIO
from zipfile import ZipFile


def test_campaign_round_trip_uses_atomic_local_record(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign_storage, "CAMPAIGN_DIR", tmp_path / "campaigns")
    campaign = create_campaign("H0 sweep", "cartesian", [{"H0": 67.0}])
    campaign.members[0].status = JobStatus.COMPLETED
    campaign.members[0].sigma8 = 0.81
    campaign.members[0].evidence = {"class_status": "CLASS", "class_settings": {"H0": 67.0}}
    campaign.metadata = {"backend": "CLASS", "worker_count": 2}

    path = campaign_storage.save_campaign(campaign)
    restored = campaign_storage.load_campaign(campaign.campaign_id)

    assert path.exists()
    assert restored.campaign_id == campaign.campaign_id
    assert restored.members[0].status == JobStatus.COMPLETED
    assert restored.members[0].sigma8 == 0.81
    assert restored.members[0].evidence["class_settings"]["H0"] == 67.0
    assert campaign_storage.list_campaign_ids() == [campaign.campaign_id]


def test_campaign_requeues_interrupted_running_member(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign_storage, "CAMPAIGN_DIR", tmp_path / "campaigns")
    campaign = create_campaign("Interrupted", "cartesian", [{"H0": 67.0}])
    campaign.members[0].status = JobStatus.RUNNING
    campaign.metadata["lifecycle_state"] = "running"
    campaign_storage.save_campaign(campaign)

    recovered = campaign_storage.load_campaign(campaign.campaign_id)
    assert recovered.members[0].status == JobStatus.PENDING
    assert recovered.metadata["lifecycle_state"] == "pending"
    assert "requeued" in recovered.metadata["recovery_note"]


def test_campaign_bundle_contains_manifest_table_and_member_evidence():
    campaign = create_campaign("Bundle", "cartesian", [{"H0": 67.0}])
    campaign.members[0].status = JobStatus.COMPLETED
    campaign.members[0].evidence = {"class_status": "CLASS"}
    payload = campaign_storage.campaign_export_bundle(campaign)

    with ZipFile(BytesIO(payload)) as archive:
        assert set(archive.namelist()) == {
            "campaign.json",
            "members.csv",
            "README.txt",
            f"evidence/{campaign.members[0].member_id}.json",
        }
        assert "CLASS" in archive.read("members.csv").decode()


def test_completed_member_timings_ignores_failed_or_zero_members(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign_storage, "CAMPAIGN_DIR", tmp_path / "campaigns")
    campaign = create_campaign("Timings", "cartesian", [{"H0": 67.0}, {"H0": 68.0}])
    campaign.members[0].status = JobStatus.COMPLETED
    campaign.members[0].elapsed_seconds = 12.5
    campaign.members[1].status = JobStatus.FAILED
    campaign.members[1].elapsed_seconds = 99.0
    campaign_storage.save_campaign(campaign)
    assert campaign_storage.completed_member_timings() == [12.5]

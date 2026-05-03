"""
Unit tests for app.data — in-memory (USE_MOCK_DATA=true) code paths.

All tests force the mock-data path so no PostgreSQL connection is needed.
The PATIENTS / APPOINTMENTS module-level lists are saved and restored around
every test that mutates them (confirm / cancel update status in-place).
"""
import copy
import pytest
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def force_mock_data_and_reset():
    """
    1. Force USE_MOCK_DATA=True so tests never touch the real database.
    2. Snapshot the PATIENTS/APPOINTMENTS lists before each test and restore
       them afterwards so mutations from confirm/cancel don't leak between
       tests.
    """
    import app.data as data_mod

    original_patients = copy.deepcopy(data_mod.PATIENTS)
    original_appointments = copy.deepcopy(data_mod.APPOINTMENTS)
    original_flag = data_mod.USE_MOCK_DATA

    data_mod.USE_MOCK_DATA = True

    yield

    # Restore list contents (in-place so other module references stay valid)
    data_mod.PATIENTS.clear()
    data_mod.PATIENTS.extend(original_patients)
    data_mod.APPOINTMENTS.clear()
    data_mod.APPOINTMENTS.extend(original_appointments)
    data_mod.USE_MOCK_DATA = original_flag


# ---------------------------------------------------------------------------
# find_patient
# ---------------------------------------------------------------------------


class TestFindPatient:
    """app.data.find_patient with in-memory data."""

    def test_returns_patient_on_exact_match(self):
        from app.data import find_patient

        result = find_patient("Alice Johnson", "555-1234", "1985-03-15")

        assert result is not None
        assert result["id"] == "P001"
        assert result["full_name"] == "Alice Johnson"

    def test_returns_none_when_wrong_name(self):
        from app.data import find_patient

        result = find_patient("Nobody Here", "555-1234", "1985-03-15")
        assert result is None

    def test_returns_none_when_wrong_phone(self):
        from app.data import find_patient

        result = find_patient("Alice Johnson", "000-0000", "1985-03-15")
        assert result is None

    def test_returns_none_when_wrong_dob(self):
        from app.data import find_patient

        result = find_patient("Alice Johnson", "555-1234", "2000-01-01")
        assert result is None

    def test_name_match_is_case_insensitive(self):
        from app.data import find_patient

        result = find_patient("alice johnson", "555-1234", "1985-03-15")
        assert result is not None
        assert result["id"] == "P001"

    def test_phone_match_ignores_dashes(self):
        """Phone normalization: '5551234' should match '555-1234'."""
        from app.data import find_patient

        result = find_patient("Alice Johnson", "5551234", "1985-03-15")
        assert result is not None
        assert result["id"] == "P001"

    def test_second_patient_can_be_found(self):
        from app.data import find_patient

        result = find_patient("Bob Smith", "555-5678", "1990-07-22")
        assert result is not None
        assert result["id"] == "P002"


# ---------------------------------------------------------------------------
# get_appointments
# ---------------------------------------------------------------------------


class TestGetAppointments:
    """app.data.get_appointments with in-memory data."""

    def test_returns_list_for_known_patient(self):
        from app.data import get_appointments

        appointments = get_appointments("P001")
        assert isinstance(appointments, list)
        assert len(appointments) >= 1

    def test_appointment_ids_belong_to_patient(self):
        from app.data import get_appointments

        appointments = get_appointments("P001")
        for appt in appointments:
            assert appt["patient_id"] == "P001"

    def test_alice_has_two_appointments(self):
        from app.data import get_appointments

        appointments = get_appointments("P001")
        ids = {a["id"] for a in appointments}
        assert "A001" in ids
        assert "A002" in ids

    def test_returns_empty_list_for_unknown_patient(self):
        from app.data import get_appointments

        appointments = get_appointments("P999")
        assert appointments == []

    def test_patients_cannot_see_each_others_appointments(self):
        from app.data import get_appointments

        alice_ids = {a["id"] for a in get_appointments("P001")}
        bob_ids = {a["id"] for a in get_appointments("P002")}
        assert alice_ids.isdisjoint(bob_ids)


# ---------------------------------------------------------------------------
# get_appointment_by_id
# ---------------------------------------------------------------------------


class TestGetAppointmentById:
    """app.data.get_appointment_by_id with in-memory data."""

    def test_returns_appointment_when_patient_matches(self):
        from app.data import get_appointment_by_id

        appt = get_appointment_by_id("A001", "P001")
        assert appt is not None
        assert appt["id"] == "A001"
        assert appt["patient_id"] == "P001"

    def test_returns_none_when_wrong_patient(self):
        """A001 belongs to P001 — P002 must not access it."""
        from app.data import get_appointment_by_id

        appt = get_appointment_by_id("A001", "P002")
        assert appt is None

    def test_returns_none_for_unknown_appointment_id(self):
        from app.data import get_appointment_by_id

        appt = get_appointment_by_id("A999", "P001")
        assert appt is None

    def test_appointment_dict_contains_expected_keys(self):
        from app.data import get_appointment_by_id

        appt = get_appointment_by_id("A001", "P001")
        for key in ("id", "patient_id", "date", "time", "doctor", "specialty", "status"):
            assert key in appt


# ---------------------------------------------------------------------------
# confirm_appointment
# ---------------------------------------------------------------------------


class TestConfirmAppointment:
    """app.data.confirm_appointment with in-memory data."""

    def test_sets_status_to_confirmed(self):
        from app.data import confirm_appointment

        result = confirm_appointment("A001", "P001")
        assert result is not None
        assert result["status"] == "confirmed"

    def test_returns_none_for_wrong_patient(self):
        from app.data import confirm_appointment

        result = confirm_appointment("A001", "P002")  # A001 belongs to P001
        assert result is None

    def test_returns_none_for_unknown_appointment(self):
        from app.data import confirm_appointment

        result = confirm_appointment("A999", "P001")
        assert result is None

    def test_returns_appointment_dict_on_success(self):
        from app.data import confirm_appointment

        result = confirm_appointment("A002", "P001")
        assert isinstance(result, dict)
        assert result["id"] == "A002"

    def test_in_memory_mutation_is_visible(self):
        """The status change must be visible to subsequent data lookups."""
        from app.data import confirm_appointment, get_appointment_by_id

        confirm_appointment("A001", "P001")
        appt = get_appointment_by_id("A001", "P001")
        assert appt["status"] == "confirmed"


# ---------------------------------------------------------------------------
# cancel_appointment
# ---------------------------------------------------------------------------


class TestCancelAppointment:
    """app.data.cancel_appointment with in-memory data."""

    def test_sets_status_to_cancelled(self):
        from app.data import cancel_appointment

        result = cancel_appointment("A001", "P001")
        assert result is not None
        assert result["status"] == "cancelled"

    def test_returns_none_for_wrong_patient(self):
        from app.data import cancel_appointment

        result = cancel_appointment("A003", "P001")  # A003 belongs to P002
        assert result is None

    def test_returns_none_for_unknown_appointment(self):
        from app.data import cancel_appointment

        result = cancel_appointment("A999", "P001")
        assert result is None

    def test_returns_appointment_dict_on_success(self):
        from app.data import cancel_appointment

        result = cancel_appointment("A002", "P001")
        assert isinstance(result, dict)
        assert result["id"] == "A002"

    def test_in_memory_mutation_is_visible(self):
        """The status change must be visible to subsequent data lookups."""
        from app.data import cancel_appointment, get_appointment_by_id

        cancel_appointment("A002", "P001")
        appt = get_appointment_by_id("A002", "P001")
        assert appt["status"] == "cancelled"


# ---------------------------------------------------------------------------
# initialize_mock_data (no-op when USE_MOCK_DATA=true)
# ---------------------------------------------------------------------------


class TestInitializeMockData:
    """initialize_mock_data must be a no-op when USE_MOCK_DATA is True."""

    def test_no_op_when_use_mock_data_true(self):
        """Should complete without errors and not interact with the DB."""
        from app.data import initialize_mock_data

        # Simply calling it must not raise — DB is not available in unit tests
        initialize_mock_data()


# ---------------------------------------------------------------------------
# get_all_patients
# ---------------------------------------------------------------------------


class TestGetAllPatients:
    """app.data.get_all_patients with in-memory data."""

    def test_returns_list(self):
        from app.data import get_all_patients

        patients = get_all_patients()
        assert isinstance(patients, list)

    def test_returns_all_seeded_patients(self):
        from app.data import get_all_patients

        patients = get_all_patients()
        ids = {p["id"] for p in patients}
        assert "P001" in ids
        assert "P002" in ids
        assert "P003" in ids

    def test_each_patient_has_required_keys(self):
        from app.data import get_all_patients

        for patient in get_all_patients():
            for key in ("id", "full_name", "phone", "date_of_birth"):
                assert key in patient

import unittest
import zipfile
from unittest.mock import patch
from io import BytesIO
from pathlib import Path

from tests.mock_tabpfn_server import with_mock_server
from tabpfn_client.service_wrapper import UserAuthenticationClient, UserDataClient
from tabpfn_client.client import ServiceClient


class TestUserAuthClient(unittest.TestCase):
    """
    These test cases are meant to validate the interface between the client and the server.
    They do not guarantee if the response from the server is correct.
    """

    def setUp(self):
        ServiceClient.reset_authorization()

    def tearDown(self):
        UserAuthenticationClient.CACHED_TOKEN_FILE.unlink(missing_ok=True)

    @with_mock_server()
    def test_set_token_by_valid_login(self, mock_server):
        # mock valid login response
        dummy_token = "dummy_token"
        mock_server.router.post(mock_server.endpoints.login.path).respond(
            200, json={"access_token": dummy_token}
        )

        self.assertTrue(
            UserAuthenticationClient.set_token_by_login(
                "dummy_email", "dummy_password"
            )[0]
        )

        # assert token is set
        self.assertEqual(dummy_token, ServiceClient.get_access_token())

    @with_mock_server()
    def test_set_token_by_invalid_login(self, mock_server):
        # mock invalid login response
        mock_server.router.post(mock_server.endpoints.login.path).respond(
            401, json={"detail": "Incorrect email or password"}
        )
        self.assertEqual(
            (False, "Incorrect email or password", 401),
            UserAuthenticationClient.set_token_by_login(
                "dummy_email", "dummy_password"
            ),
        )

        # assert token is not set
        self.assertIsNone(ServiceClient.get_access_token())

    @with_mock_server()
    def test_try_reusing_existing_token(self, mock_server):
        # create dummy token file
        dummy_token = "dummy_token"
        token_file = UserAuthenticationClient.CACHED_TOKEN_FILE
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text(dummy_token)

        # mock authentication
        mock_server.router.get(mock_server.endpoints.protected_root.path).respond(200)

        # assert no exception is raised
        UserAuthenticationClient.try_reuse_existing_token()

        # assert token is set
        self.assertEqual(dummy_token, ServiceClient.get_access_token())

    def test_try_reusing_non_existing_token(self):
        # assert no exception is raised
        UserAuthenticationClient.try_reuse_existing_token()

        # assert token is not set
        self.assertIsNone(ServiceClient.get_access_token())

    @with_mock_server()
    def test_set_token_by_invalid_registration(self, mock_server):
        # mock invalid registration response
        mock_server.router.post(mock_server.endpoints.register.path).respond(
            401, json={"detail": "Password mismatch"}
        )
        self.assertEqual(
            (False, "Password mismatch", None),
            UserAuthenticationClient.set_token_by_registration(
                "dummy_email",
                "dummy_password",
                "dummy_password",
                "dummy_validation",
                {
                    "company": "dummy_company",
                    "use_case": "dummy_usecase",
                    "role": "dummy_role",
                    "contact_via_email": False,
                },
            ),
        )

        # assert token is not set
        self.assertIsNone(ServiceClient.get_access_token())

    @with_mock_server()
    def test_reset_cache_after_token_set(self, mock_server):
        # set token from a dummy file
        dummy_token = "dummy_token"
        token_file = UserAuthenticationClient.CACHED_TOKEN_FILE
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text(dummy_token)

        # mock authentication
        mock_server.router.get(mock_server.endpoints.protected_root.path).respond(200)
        self.assertTrue(UserAuthenticationClient.try_reuse_existing_token()[0])

        # assert token is set
        self.assertEqual(dummy_token, ServiceClient.get_access_token())

        # reset cache
        UserAuthenticationClient.reset_cache()

        # assert token is not set
        self.assertIsNone(ServiceClient.get_access_token())

    def test_reset_cache_without_token_set(self):
        # assert no exception is raised
        UserAuthenticationClient.reset_cache()

        # assert token is not set
        self.assertIsNone(ServiceClient.get_access_token())


class TestUserDataClient(unittest.TestCase):
    """
    These test cases are meant to validate the interface between the client and the server.
    They do not guarantee if the response from the server is correct.
    """

    @staticmethod
    def _is_zip_file_empty(zip_file_path: Path):
        return not zipfile.ZipFile(zip_file_path, "r").namelist()

    @with_mock_server()
    def test_get_data_summary_accepts_dict(self, mock_server):
        # mock get_data_summary response
        mock_summary = {
            "content": "does not matter as long as this is returned by the server"
        }
        mock_server.router.get(mock_server.endpoints.get_data_summary.path).respond(
            200, json=mock_summary
        )

        self.assertEqual(mock_summary, UserDataClient.get_data_summary())

    @with_mock_server()
    def test_download_all_data_accepts_empty_zip(self, mock_server):
        # mock download_all_data response (with empty zip file)
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w"):
            pass
        zip_buffer.seek(0)

        mock_server.router.get(mock_server.endpoints.download_all_data.path).respond(
            200,
            stream=zip_buffer,
            headers={"Content-Disposition": "attachment; filename=all_data.zip"},
        )

        # assert no exception is raised, and zip file is empty
        zip_file_path = UserDataClient.download_all_data(Path("."))
        self.assertTrue(self._is_zip_file_empty(zip_file_path))

        # delete the zip file
        zip_file_path.unlink()

    @with_mock_server()
    def test_download_all_data_accepts_non_empty_zip(self, mock_server):
        # mock download_all_data response (with non-empty zip file)
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            zip_file.writestr("dummy_file.txt", "dummy content")
        zip_buffer.seek(0)

        mock_server.router.get(mock_server.endpoints.download_all_data.path).respond(
            200,
            stream=zip_buffer,
            headers={"Content-Disposition": "attachment; filename=all_data.zip"},
        )

        # assert no exception is raised, and zip file is not empty
        zip_file_path = UserDataClient.download_all_data(Path("."))
        self.assertFalse(self._is_zip_file_empty(zip_file_path))

        # delete the zip file
        zip_file_path.unlink()

    @with_mock_server()
    def test_delete_datasets_accepts_empty_uid_list(self, mock_server):
        # mock delete_dataset response (with empty list)
        mock_server.router.delete(mock_server.endpoints.delete_dataset.path).respond(
            200, json={"deleted_dataset_uids": []}
        )

        # assert no exception is raised
        self.assertEqual([], UserDataClient.delete_dataset("dummy_uid"))

    @with_mock_server()
    def test_delete_datasets_accepts_uid_list(self, mock_server):
        # mock delete_dataset response (with non-empty list)
        mock_server.router.delete(mock_server.endpoints.delete_dataset.path).respond(
            200, json={"deleted_dataset_uids": ["dummy_uid"]}
        )

        # assert no exception is raised
        self.assertEqual(["dummy_uid"], UserDataClient.delete_dataset("dummy_uid"))

    @with_mock_server()
    def test_delete_all_datasets_accepts_empty_uid_list(self, mock_server):
        # mock delete_all_datasets response (with empty list)
        mock_server.router.delete(
            mock_server.endpoints.delete_all_datasets.path
        ).respond(200, json={"deleted_dataset_uids": []})

        # assert no exception is raised
        self.assertEqual([], UserDataClient.delete_all_datasets())

    @with_mock_server()
    def test_delete_all_datasets_accepts_uid_list(self, mock_server):
        # mock delete_all_datasets response (with non-empty list)
        mock_server.router.delete(
            mock_server.endpoints.delete_all_datasets.path
        ).respond(200, json={"deleted_dataset_uids": ["dummy_uid"]})

        # assert no exception is raised
        self.assertEqual(["dummy_uid"], UserDataClient.delete_all_datasets())

    @with_mock_server()
    @patch(
        "tabpfn_client.prompt_agent.PromptAgent.prompt_confirm_password_for_user_account_deletion"
    )
    def test_delete_user_account_with_valid_password(
        self, mock_server, mock_prompt_confirm_password
    ):
        # mock delete_user_account response
        mock_server.router.delete(
            mock_server.endpoints.delete_user_account.path
        ).respond(200)

        # mock password prompting
        mock_prompt_confirm_password.return_value = "dummy_password"

        # assert no exception is raised
        UserDataClient.delete_user_account()

    @with_mock_server()
    @patch(
        "tabpfn_client.prompt_agent.PromptAgent.prompt_confirm_password_for_user_account_deletion"
    )
    def test_delete_user_account_with_invalid_password(
        self, mock_server, mock_prompt_confirm_password
    ):
        # mock delete_user_account response
        mock_server.router.delete(
            mock_server.endpoints.delete_user_account.path
        ).respond(400)

        # mock password prompting
        mock_prompt_confirm_password.return_value = "dummy_password"

        # assert exception is raised
        self.assertRaises(RuntimeError, UserDataClient.delete_user_account)

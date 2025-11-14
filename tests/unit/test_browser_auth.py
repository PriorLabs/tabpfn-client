import unittest
from unittest.mock import patch, MagicMock
from tabpfn_client.browser_auth import BrowserAuthHandler
import time
import threading
import urllib.parse
import http.client
import socketserver


class TestBrowserAuthHandler(unittest.TestCase):
    FALLBACK_MESSAGE = "\nCould not open browser automatically. Falling back to command-line login...\n"

    @patch("tabpfn_client.browser_auth.socketserver.TCPServer")
    @patch("builtins.print")
    def test_server_setup_exception(self, mock_print, mock_tcp_server):
        # Simulate exception during callback server setup
        mock_tcp_server.side_effect = Exception("Server setup failed")

        # Instantiate BrowserAuthHandler
        browser_auth = BrowserAuthHandler(gui_url="http://example.com")

        # Call try_browser_login
        success, token = browser_auth.try_browser_login()

        # Assert that the method returned failure
        self.assertFalse(success)
        self.assertIsNone(token)

    @patch("tabpfn_client.browser_auth.webbrowser.open", return_value=False)
    @patch("builtins.print")
    def test_browser_open_failure(self, mock_print, mock_webbrowser_open):
        # Instantiate BrowserAuthHandler
        browser_auth = BrowserAuthHandler(gui_url="http://example.com")

        # Call try_browser_login
        success, token = browser_auth.try_browser_login()

        # Assert that browser open was attempted
        mock_webbrowser_open.assert_called_once()

        # Assert that the method returned failure
        self.assertFalse(success)
        self.assertIsNone(token)

        # Verify that fallback message was printed
        mock_print.assert_any_call(self.FALLBACK_MESSAGE)

    @patch("tabpfn_client.browser_auth.webbrowser.open", return_value=True)
    @patch("tabpfn_client.browser_auth.socketserver.TCPServer")
    @patch("tabpfn_client.browser_auth.Event")
    @patch("builtins.print")
    def test_user_cancels_login(
        self, mock_event, mock_tcp_server, mock_webbrowser_open, mock_print
    ):
        # Mock HTTP server and auth event
        mock_httpd = MagicMock()
        mock_tcp_server.return_value.__enter__.return_value = mock_httpd

        # Simulate the auth event never being set
        mock_auth_event = MagicMock()
        mock_auth_event.is_set.return_value = False
        mock_event.return_value = mock_auth_event

        # Set up `handle_request` to raise an exception after a few calls to break the loop
        handle_request_call_count = {"count": 0}

        def handle_request_side_effect():
            handle_request_call_count["count"] += 1
            if handle_request_call_count["count"] > 2:
                raise TimeoutError("Simulated timeout")

        mock_httpd.handle_request.side_effect = handle_request_side_effect

        # Instantiate BrowserAuthHandler
        browser_auth = BrowserAuthHandler(gui_url="http://example.com")

        # Call try_browser_login
        success, token = browser_auth.try_browser_login()

        # Assert that the method returned failure
        self.assertFalse(success)
        self.assertIsNone(token)

    def simulate_callback(self, mock_webbrowser, test_token):
        """Helper method to simulate the callback response"""
        time.sleep(0.1)  # Give the server time to start
        calls = mock_webbrowser.call_args_list
        login_url = calls[0][0][0]
        callback_url = urllib.parse.parse_qs(urllib.parse.urlparse(login_url).query)[
            "callback"
        ][0]
        port = int(callback_url.split(":")[-1])

        conn = http.client.HTTPConnection("localhost", port)
        conn.request("GET", f"/?token={test_token}")
        response = conn.getresponse()
        html = response.read().decode()
        conn.close()
        return login_url, html

    @patch("webbrowser.open")
    def test_successful_browser_login(self, mock_webbrowser):
        mock_webbrowser.return_value = True
        test_token = "test_token_123"

        # Start callback simulation thread
        callback_thread = threading.Thread(
            target=lambda: self.simulate_callback(mock_webbrowser, test_token)
        )
        callback_thread.daemon = True
        callback_thread.start()

        browser_auth = BrowserAuthHandler(gui_url="http://example.com")

        # Call try_browser_login
        success, token = browser_auth.try_browser_login()

        # Verify results
        self.assertTrue(success)
        self.assertEqual(token, test_token)
        self.assertTrue(mock_webbrowser.called)

        # Verify URL format
        login_url = mock_webbrowser.call_args[0][0]
        parsed_url = urllib.parse.urlparse(login_url)
        self.assertEqual(parsed_url.scheme, "http")
        self.assertEqual(parsed_url.netloc, "example.com")
        self.assertEqual(parsed_url.path, "/login")
        self.assertTrue("callback" in urllib.parse.parse_qs(parsed_url.query))

        # Let callback thread finish
        callback_thread.join(timeout=1)

    @patch("webbrowser.open")
    def test_invalid_token_response(self, mock_webbrowser):
        """Test handling of invalid/empty token in callback"""
        mock_webbrowser.return_value = True

        # Start callback simulation with empty token
        callback_thread = threading.Thread(
            target=lambda: self.simulate_callback(mock_webbrowser, "")
        )
        callback_thread.daemon = True
        callback_thread.start()

        browser_auth = BrowserAuthHandler(gui_url="http://example.com")

        success, token = browser_auth.try_browser_login()
        self.assertFalse(success)
        self.assertIsNone(token)
        callback_thread.join(timeout=1)

    @patch("webbrowser.open")
    def test_multiple_callback_requests(self, mock_webbrowser):
        """Test handling of multiple callback requests"""
        mock_webbrowser.return_value = True
        test_token = "test_token_123"

        def simulate_multiple_callbacks():
            try:
                # First callback with one token
                self.simulate_callback(mock_webbrowser, test_token)
                # Second callback with different token - should be ignored
                # This will fail with ConnectionRefused, which is expected
                self.simulate_callback(mock_webbrowser, "different_token")
            except ConnectionRefusedError:
                # This is expected as the server stops after first successful callback
                pass

        callback_thread = threading.Thread(target=simulate_multiple_callbacks)
        callback_thread.daemon = True
        callback_thread.start()

        browser_auth = BrowserAuthHandler(gui_url="http://example.com")

        success, token = browser_auth.try_browser_login()
        self.assertTrue(success)
        self.assertEqual(token, test_token)  # Should use first token
        callback_thread.join(timeout=1)

    @patch("webbrowser.open")
    @patch("tabpfn_client.browser_auth.socketserver.TCPServer")
    def test_timeout_handling(self, mock_tcp_server, mock_webbrowser):
        """Test handling of timeout during login"""
        mock_webbrowser.return_value = True

        # Mock HTTP server
        mock_httpd = MagicMock()
        mock_tcp_server.return_value.__enter__.return_value = mock_httpd

        # Simulate timeout by making handle_request raise TimeoutError
        mock_httpd.handle_request.side_effect = TimeoutError("Request timed out")

        browser_auth = BrowserAuthHandler(gui_url="http://example.com")
        success, token = browser_auth.try_browser_login()

        self.assertFalse(success)
        self.assertIsNone(token)

    @patch("webbrowser.open")
    def test_malformed_callback_url(self, mock_webbrowser):
        """Test handling of malformed callback URL"""
        mock_webbrowser.return_value = True

        def simulate_malformed_request():
            time.sleep(0.1)
            calls = mock_webbrowser.call_args_list
            callback_url = urllib.parse.parse_qs(
                urllib.parse.urlparse(calls[0][0][0]).query
            )["callback"][0]
            port = int(callback_url.split(":")[-1])

            # Send malformed request
            conn = http.client.HTTPConnection("localhost", port)
            conn.request("GET", "/malformed?not_a_token=123")
            conn.getresponse()
            conn.close()

        callback_thread = threading.Thread(target=simulate_malformed_request)
        callback_thread.daemon = True
        callback_thread.start()

        browser_auth = BrowserAuthHandler(gui_url="http://example.com")
        success, token = browser_auth.try_browser_login()
        self.assertFalse(success)
        self.assertIsNone(token)
        callback_thread.join(timeout=1)

    @patch("webbrowser.open")
    def test_server_port_in_use(self, mock_webbrowser):
        """Test handling when preferred port is in use"""
        mock_webbrowser.return_value = True

        # Create a server to occupy a port
        with socketserver.TCPServer(
            ("", 0), http.server.SimpleHTTPRequestHandler
        ) as blocking_server:
            port = blocking_server.server_address[1]

            # Try to start auth server on same port
            with patch("socketserver.TCPServer.__init__") as mock_server:
                mock_server.side_effect = OSError(f"Port {port} already in use")

                browser_auth = BrowserAuthHandler(gui_url="http://example.com")
                success, token = browser_auth.try_browser_login()
                self.assertFalse(success)
                self.assertIsNone(token)

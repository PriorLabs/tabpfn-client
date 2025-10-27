#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

from threading import Event
import http.server
import socketserver
import webbrowser
import urllib.parse
from typing import Optional, Tuple


class BrowserAuthHandler:
    def __init__(self, gui_url: str):
        self.gui_url = gui_url

    def try_browser_login(self) -> Tuple[bool, Optional[str]]:
        """
        Attempts to perform browser-based login
        Returns (success: bool, token: Optional[str])
        """
        auth_event = Event()
        received_token = None
        gui_url = self.gui_url

        class CallbackHandler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                nonlocal received_token
                nonlocal gui_url

                parsed = urllib.parse.urlparse(self.path)
                query = urllib.parse.parse_qs(parsed.query)

                if "token" in query:
                    received_token = query["token"][0]

                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                success_html = f"""
                <html>
                    <body style="text-align: center; font-family: Arial, sans-serif; padding: 50px;">
                        <h2>Login successful!</h2>
                        <p>You can close this window and return to your application.</p>
                        <script>
                            window.location.href = "{gui_url}/redirect-success";
                        </script>
                    </body>
                </html>
                """
                self.wfile.write(success_html.encode())
                auth_event.set()

            def log_message(self, format, *args):
                pass

        try:
            with socketserver.TCPServer(("", 0), CallbackHandler) as httpd:
                port = httpd.server_address[1]
                callback_url = f"http://localhost:{port}"

                login_url = f"{self.gui_url}/login?callback={callback_url}"

                print(
                    "\nOpening browser for login. Please complete the login/registration process in your browser and return here.\n"
                )

                if not webbrowser.open(login_url):
                    print(
                        "\nCould not open browser automatically. Falling back to command-line login...\n"
                    )
                    return False, None

                while not auth_event.is_set():
                    httpd.handle_request()

                return received_token is not None, received_token

        except Exception:
            print("\n Browser auth failed. Falling back to command-line login...\n")
            return False, None

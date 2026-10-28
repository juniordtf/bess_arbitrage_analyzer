import ssl
import sys

import requests


class _WindowsTrustAdapter(requests.adapters.HTTPAdapter):
    def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
        import truststore

        pool_kwargs["ssl_context"] = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        super().init_poolmanager(
            connections, maxsize, block=block, **pool_kwargs
        )


def use_system_certificates(session: requests.Session) -> None:
    if sys.platform == "win32" and sys.version_info >= (3, 10):
        session.mount("https://", _WindowsTrustAdapter())

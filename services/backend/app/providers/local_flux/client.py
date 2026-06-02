from pathlib import Path
from typing import Any

import httpx

from app.settings import Settings


class LocalFluxClientError(RuntimeError):
    pass


class LocalFluxClient:
    def __init__(self, settings: Settings):
        self.base_url = settings.local_flux_base_url.rstrip("/")
        self.timeout = settings.local_flux_timeout_seconds

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def get_system_stats(self) -> dict[str, Any]:
        try:
            with httpx.Client(timeout=5) as client:
                response = client.get(self._url("/system_stats"))
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise LocalFluxClientError(str(error)) from error
        if not isinstance(payload, dict):
            raise LocalFluxClientError("Local Flux status returned an invalid payload.")
        return payload

    def get_models(self, folder: str) -> list[str]:
        try:
            with httpx.Client(timeout=10) as client:
                response = client.get(self._url(f"/models/{folder}"))
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise LocalFluxClientError(str(error)) from error
        if not isinstance(payload, list):
            raise LocalFluxClientError("Local Flux models endpoint returned an invalid payload.")
        return [str(item) for item in payload]

    def upload_image(self, image_path: Path) -> str:
        try:
            with image_path.open("rb") as file_obj:
                files = {"image": (image_path.name, file_obj, "application/octet-stream")}
                data = {"overwrite": "true"}
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(self._url("/upload/image"), files=files, data=data)
                    response.raise_for_status()
                    payload = response.json()
        except (OSError, httpx.HTTPError, ValueError) as error:
            raise LocalFluxClientError(str(error)) from error
        if not isinstance(payload, dict):
            raise LocalFluxClientError("Local Flux upload returned an invalid payload.")
        filename = payload.get("name") or payload.get("filename")
        if not isinstance(filename, str) or not filename.strip():
            raise LocalFluxClientError("Local Flux upload did not return an image filename.")
        return filename

    def post_prompt(self, prompt: dict[str, Any], client_id: str) -> str:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    self._url("/prompt"),
                    json={"prompt": prompt, "client_id": client_id},
                )
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise LocalFluxClientError(str(error)) from error
        if not isinstance(payload, dict):
            raise LocalFluxClientError("Local Flux prompt endpoint returned an invalid payload.")
        prompt_id = payload.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise LocalFluxClientError("Local Flux prompt endpoint did not return prompt_id.")
        return prompt_id

    def get_history(self, prompt_id: str) -> dict[str, Any]:
        try:
            with httpx.Client(timeout=20) as client:
                response = client.get(self._url(f"/history/{prompt_id}"))
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise LocalFluxClientError(str(error)) from error
        if not isinstance(payload, dict):
            raise LocalFluxClientError("Local Flux history endpoint returned an invalid payload.")
        return payload

    def view_image(self, *, filename: str, subfolder: str = "", image_type: str = "output") -> bytes:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(
                    self._url("/view"),
                    params={"filename": filename, "subfolder": subfolder, "type": image_type},
                )
                response.raise_for_status()
                content = response.content
        except httpx.HTTPError as error:
            raise LocalFluxClientError(str(error)) from error
        if not content:
            raise LocalFluxClientError("Local Flux returned an empty image.")
        return content

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

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.is_success:
            return
        detail = _response_error_detail(response)
        raise LocalFluxClientError(
            f"HTTP {response.status_code} {response.reason_phrase} for {response.url}: {detail}"
        )

    def get_system_stats(self) -> dict[str, Any]:
        try:
            with httpx.Client(timeout=5) as client:
                response = client.get(self._url("/system_stats"))
                self._raise_for_status(response)
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
                self._raise_for_status(response)
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
                    self._raise_for_status(response)
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
                self._raise_for_status(response)
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
                self._raise_for_status(response)
                payload = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise LocalFluxClientError(str(error)) from error
        if not isinstance(payload, dict):
            raise LocalFluxClientError("Local Flux history endpoint returned an invalid payload.")
        return payload

    def free_memory(self) -> None:
        try:
            with httpx.Client(timeout=10) as client:
                response = client.post(
                    self._url("/free"),
                    json={"unload_models": True, "free_memory": True},
                )
                self._raise_for_status(response)
        except (httpx.HTTPError, ValueError) as error:
            raise LocalFluxClientError(str(error)) from error

    def view_image(self, *, filename: str, subfolder: str = "", image_type: str = "output") -> bytes:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(
                    self._url("/view"),
                    params={"filename": filename, "subfolder": subfolder, "type": image_type},
                )
                self._raise_for_status(response)
                content = response.content
        except httpx.HTTPError as error:
            raise LocalFluxClientError(str(error)) from error
        if not content:
            raise LocalFluxClientError("Local Flux returned an empty image.")
        return content


def _response_error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return _trim(response.text)
    if not isinstance(payload, dict):
        return _trim(str(payload))

    parts: list[str] = []
    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        details = error.get("details")
        if isinstance(message, str) and message:
            parts.append(message)
        if isinstance(details, str) and details:
            parts.append(details)
    elif isinstance(error, str) and error:
        parts.append(error)

    node_errors = payload.get("node_errors")
    if isinstance(node_errors, dict) and node_errors:
        formatted_nodes = []
        for node_id, node_error in list(node_errors.items())[:5]:
            if isinstance(node_error, dict):
                class_type = node_error.get("class_type")
                node_message = node_error.get("message")
                errors = node_error.get("errors")
                node_parts = [str(node_id)]
                if isinstance(class_type, str) and class_type:
                    node_parts.append(class_type)
                if isinstance(node_message, str) and node_message:
                    node_parts.append(node_message)
                if isinstance(errors, list) and errors:
                    first_error = errors[0]
                    if isinstance(first_error, dict):
                        message = first_error.get("message")
                        details = first_error.get("details")
                        if isinstance(message, str) and message:
                            node_parts.append(message)
                        if isinstance(details, str) and details:
                            node_parts.append(details)
                formatted_nodes.append(": ".join(node_parts))
        if formatted_nodes:
            parts.append("node_errors: " + " | ".join(formatted_nodes))

    if parts:
        return _trim(" ".join(parts))
    return _trim(str(payload))


def _trim(value: str, limit: int = 1400) -> str:
    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}..."

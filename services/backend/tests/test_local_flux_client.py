import httpx

from app.providers.local_flux.client import LocalFluxClient, LocalFluxClientError
from app.settings import Settings


def test_post_prompt_includes_comfy_node_errors(monkeypatch) -> None:
    def fake_post(self, url, json):
        return httpx.Response(
            400,
            request=httpx.Request("POST", url),
            json={
                "error": {
                    "message": "Prompt outputs failed validation",
                    "details": "invalid prompt",
                },
                "node_errors": {
                    "195": {
                        "class_type": "CLIPLoader",
                        "message": "Value not in list",
                        "errors": [
                            {
                                "message": "clip_name",
                                "details": "qwen_3_8b_fp8mixed.safetensors",
                            }
                        ],
                    }
                },
            },
        )

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    client = LocalFluxClient(Settings(_env_file=None))

    try:
        client.post_prompt({"1": {"class_type": "SaveImage", "inputs": {}}}, "job_test")
    except LocalFluxClientError as error:
        message = str(error)
    else:
        raise AssertionError("Expected LocalFluxClientError")

    assert "Prompt outputs failed validation" in message
    assert "CLIPLoader" in message
    assert "qwen_3_8b_fp8mixed.safetensors" in message

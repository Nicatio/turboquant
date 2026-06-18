from __future__ import annotations

import argparse

import uvicorn

from turboquant.openai_compatible_server import (
    ServerConfig,
    create_openai_compatible_app,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve local Gemma 4 as an OpenAI-compatible API for editors like Continue."
    )
    parser.add_argument("--model", default="mlx-community/gemma-4-26b-a4b-it-4bit")
    parser.add_argument("--served-model-name", default="gemma4-mlx")
    parser.add_argument(
        "--implementation",
        choices=["baseline", "shadow", "direct"],
        default="shadow",
    )
    parser.add_argument("--bits", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--recent-window", type=int, default=256)
    parser.add_argument("--recent-slack", type=int, default=8)
    parser.add_argument("--dense-shadow", action="store_true")
    parser.add_argument(
        "--share-quantizers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--prefill-step-size", type=int, default=256)
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    app = create_openai_compatible_app(
        ServerConfig(
            model_ref=args.model,
            served_model_name=args.served_model_name,
            implementation=args.implementation,
            bits=args.bits,
            seed=args.seed,
            block_size=args.block_size,
            recent_window=args.recent_window,
            recent_slack=args.recent_slack,
            dense_shadow=args.dense_shadow,
            share_quantizers=args.share_quantizers,
            prefill_step_size=args.prefill_step_size,
            api_key=args.api_key,
        )
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

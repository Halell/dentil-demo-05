from __future__ import annotations
import typer, json
from pathlib import Path
from rich.console import Console
from ..pipeline.t2_gazetteer import run_gazetteer
from ..pipeline.cand_faiss import run_vector_faiss
from ..pipeline.hybrid_ranker import merge_and_rank
from ..pipeline.mv_candidates_builder import build_mv_v2
from ..pipeline.router_block2 import run_block2_all

app = typer.Typer(help="Block 2 candidate generation pipeline")
console = Console()

@app.command("gazetteer")
def cmd_gazetteer(merged_block1: str, out: str):
    run_gazetteer(merged_block1, out)
    console.print(f"[green]Gazetteer hits saved to {out}[/green]")

@app.command("vector")
def cmd_vector(merged_block1: str, out: str):
    run_vector_faiss(merged_block1, out)
    console.print(f"[green]Vector hits saved to {out}[/green]")

@app.command("hybrid")
def cmd_hybrid(gazetteer_file: str, vector_file: str, out: str):
    merge_and_rank(gazetteer_file, vector_file, out)
    console.print(f"[green]Merged candidates saved to {out}[/green]")

@app.command("build-mv")
def cmd_build_mv(merged_candidates: str, out: str):
    build_mv_v2(merged_candidates, out)
    console.print(f"[green]MV v2 saved to {out}[/green]")

@app.command("run-all")
def cmd_run_all(merged_block1: str, out_dir: str):
    res = run_block2_all(merged_block1, out_dir)
    console.print("[green]Completed Block2 pipeline[/green]")
    for k,v in res.items():
        console.print(f" - {k}: {v}")

def main():
    app()

if __name__ == "__main__":
    main()

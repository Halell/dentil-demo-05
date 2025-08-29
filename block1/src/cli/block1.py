"""
CLI for Block 1 processing - N0/T1 normalization and tokenization.
"""
import json
import sys
from pathlib import Path
from typing import Optional, List
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import track

# Add parent to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.pipeline.n0_normalize import normalize_n0
from src.pipeline.t1_tokenize import tokenize_t1
from src.pipeline.mv_builder import build_marked_view
from src.pipeline.n0b_llm_refine import refine_with_llm
from src.pipeline.t1b_llm_hints import generate_token_hints
from src.common.schemas import ProcessedLine
from src.pipeline.ops_apply import orchestrate_block1


app = typer.Typer(help="Block 1 CLI - Dental Text Processing")
console = Console()


@app.command("run-n0-t1")
def run_n0_t1(
    input_file: str = typer.Option(..., "--in", help="Input file path"),
    output_dir: str = typer.Option(..., "--out", help="Output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Run N0 normalization and T1 tokenization on input file."""
    
    # Validate paths
    input_path = Path(input_file)
    if not input_path.exists():
        console.print(f"[red]Error: Input file {input_file} not found[/red]")
        raise typer.Exit(1)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Read input lines
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    console.print(f"[green]Processing {len(lines)} lines...[/green]")
    
    # Process each line
    n0_results = []
    t1_results = []
    
    for line in track(lines, description="Processing..."):
        # N0 normalization
        n0_result = normalize_n0(line)
        n0_results.append(n0_result.model_dump())
        
        # T1 tokenization
        t1_result = tokenize_t1(n0_result)
        t1_results.append(t1_result.model_dump())
        
        if verbose:
            console.print(f"\n[blue]Input:[/blue] {line}")
            console.print(f"[green]Normalized:[/green] {n0_result.normalized_text}")
            console.print(f"[yellow]Tokens:[/yellow] {len(t1_result.tokens)}")
    
    # Save results
    n0_output = output_path / "n0_normalized.jsonl"
    t1_output = output_path / "t1_tokens.jsonl"
    
    # Write N0 results
    with open(n0_output, 'w', encoding='utf-8') as f:
        for result in n0_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    # Write T1 results
    with open(t1_output, 'w', encoding='utf-8') as f:
        for result in t1_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    console.print(f"\n[green]Saved N0 results to {n0_output}[/green]")
    console.print(f"[green]Saved T1 results to {t1_output}[/green]")
    
    # Display summary
    display_summary(n0_results, t1_results)


@app.command("run-all")
def run_all(
    input_file: str = typer.Option(..., "--in", help="Input file path"),
    output_dir: str = typer.Option(..., "--out", help="Output directory"),
    with_llm: bool = typer.Option(False, "--llm", help="Include LLM processing"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Run complete Block 1 pipeline including optional LLM augmentation."""
    
    # First run N0+T1
    run_n0_t1(input_file, output_dir, verbose)
    
    if with_llm:
        console.print("\n[green]Running LLM augmentation...[/green]")
        llm_results = process_with_llm(output_dir, verbose)
    
    # Create merged output
    output_path = Path(output_dir)
    n0_file = output_path / "n0_normalized.jsonl"
    t1_file = output_path / "t1_tokens.jsonl"
    merged_file = output_path / "merged_block1.jsonl"
    
    # Read and merge results
    with open(n0_file, 'r', encoding='utf-8') as f:
        n0_results = [json.loads(line) for line in f]
    
    with open(t1_file, 'r', encoding='utf-8') as f:
        t1_results = [json.loads(line) for line in f]
    
    # Read LLM results if available
    llm_file = output_path / "n0b_t1b_llm_aug.jsonl"
    llm_results = []
    if llm_file.exists():
        with open(llm_file, 'r', encoding='utf-8') as f:
            llm_results = [json.loads(line) for line in f]
    
    # Merge and save
    merged_results = []
    for i, (n0, t1) in enumerate(zip(n0_results, t1_results)):
        merged = {
            "raw_text": n0["raw_text"],
            "n0": n0,
            "t1": t1,
            "llm_aug": llm_results[i] if i < len(llm_results) else None
        }
        merged_results.append(merged)
    
    with open(merged_file, 'w', encoding='utf-8') as f:
        for result in merged_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    console.print(f"[green]Saved merged results to {merged_file}[/green]")


@app.command("show-mv")
def show_marked_view(
    text: str = typer.Argument(..., help="Text to process"),
    save_to: Optional[str] = typer.Option(None, "--save", help="Save MV to file")
):
    """Generate and display Marked View for given text."""
    
    # Process text
    n0_result = normalize_n0(text)
    t1_result = tokenize_t1(n0_result)
    
    # Build MV
    mv = build_marked_view(n0_result, t1_result)
    
    # Display
    console.print("\n[blue]===== MARKED VIEW =====[/blue]")
    console.print(mv)
    console.print("[blue]======================[/blue]\n")
    
    # Save if requested
    if save_to:
        with open(save_to, 'w', encoding='utf-8') as f:
            f.write(mv)
        console.print(f"[green]Saved MV to {save_to}[/green]")


@app.command("stats")
def show_stats(
    input_file: str = typer.Argument(..., help="Processed JSONL file")
):
    """Show statistics for processed file."""
    
    path = Path(input_file)
    if not path.exists():
        console.print(f"[red]Error: File {input_file} not found[/red]")
        raise typer.Exit(1)
    
    # Read results
    with open(path, 'r', encoding='utf-8') as f:
        results = [json.loads(line) for line in f]
    
    # Calculate stats
    total_lines = len(results)
    total_tokens = 0
    total_pairs = 0
    total_numbers = 0
    
    for result in results:
        if "t1" in result:
            total_tokens += len(result["t1"]["tokens"])
        if "n0" in result:
            total_pairs += len(result["n0"]["pairs"])
            total_numbers += len(result["n0"]["numbers"])
    
    # Display stats table
    table = Table(title="Processing Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Lines", str(total_lines))
    table.add_row("Total Tokens", str(total_tokens))
    table.add_row("Avg Tokens/Line", f"{total_tokens/total_lines:.1f}" if total_lines > 0 else "0")
    table.add_row("Total Pairs", str(total_pairs))
    table.add_row("Total Numbers", str(total_numbers))
    
    console.print(table)


def process_with_llm(output_dir: str, verbose: bool = False) -> List[dict]:
    """Process results with LLM augmentation."""
    output_path = Path(output_dir)
    n0_file = output_path / "n0_normalized.jsonl"
    t1_file = output_path / "t1_tokens.jsonl"
    
    # Read results
    with open(n0_file, 'r', encoding='utf-8') as f:
        n0_results = [json.loads(line) for line in f]
    
    with open(t1_file, 'r', encoding='utf-8') as f:
        t1_results = [json.loads(line) for line in f]
    
    llm_results = []
    
    for n0_data, t1_data in track(
        zip(n0_results, t1_results), 
        description="LLM processing...",
        total=len(n0_results)
    ):
        try:
            # Reconstruct objects for processing
            from src.common.schemas import NormalizationResult, TokensResult, Token, Pair
            
            # Reconstruct N0 result
            pairs = [Pair(**pair_data) for pair_data in n0_data.get("pairs", [])]
            n0_result = NormalizationResult(
                raw_text=n0_data["raw_text"],
                normalized_text=n0_data["normalized_text"],
                numbers=n0_data.get("numbers", []),
                pairs=pairs,
                units_found=n0_data.get("units_found", []),
                dates=n0_data.get("dates", []),
                times=n0_data.get("times", []),
                notes=n0_data.get("notes", [])
            )
            
            # Reconstruct T1 result
            tokens = []
            for token_data in t1_data.get("tokens", []):
                from src.common.schemas import TokenMeta
                meta = None
                if token_data.get("meta"):
                    meta = TokenMeta(**token_data["meta"])
                
                token = Token(
                    idx=token_data["idx"],
                    text=token_data["text"],
                    kind=token_data["kind"],
                    span=token_data["span"],
                    script=token_data.get("script"),
                    meta=meta
                )
                tokens.append(token)
            
            t1_result = TokensResult(
                text=t1_data["text"],
                tokens=tokens
            )
            
            # Build MV
            mv = build_marked_view(n0_result, t1_result)
            
            # Run LLM refinement
            llm_refine_result = refine_with_llm(mv, n0_result, t1_result)
            
            # Run token hints
            llm_hints_result = generate_token_hints(mv, tokens)
            
            # Combine results
            combined_result = {
                "refinement": llm_refine_result,
                "hints": llm_hints_result,
                "mv": mv if verbose else None
            }
            
            llm_results.append(combined_result)
            
            if verbose:
                console.print(f"\n[blue]Processed:[/blue] {n0_data['raw_text']}")
                if not llm_refine_result.get("error"):
                    console.print(f"[green]Canonical terms:[/green] {llm_refine_result.get('canonical_terms', [])}")
                
        except Exception as e:
            error_result = {
                "error": str(e),
                "refinement": None,
                "hints": None
            }
            llm_results.append(error_result)
            
            if verbose:
                console.print(f"[red]Error processing line: {str(e)}[/red]")
    
    # Save LLM results
    llm_output = output_path / "n0b_t1b_llm_aug.jsonl"
    with open(llm_output, 'w', encoding='utf-8') as f:
        for result in llm_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    console.print(f"[green]Saved LLM results to {llm_output}[/green]")
    return llm_results


def display_summary(n0_results: List[dict], t1_results: List[dict]):
    """Display processing summary."""
    
    # Calculate statistics
    total_pairs = sum(len(r["pairs"]) for r in n0_results)
    total_numbers = sum(len(r["numbers"]) for r in n0_results)
    total_tokens = sum(len(r["tokens"]) for r in t1_results)
    
    # Create summary table
    table = Table(title="Processing Summary")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Count", style="green")
    
    table.add_row("Lines Processed", str(len(n0_results)))
    table.add_row("Total Pairs Found", str(total_pairs))
    table.add_row("Total Numbers", str(total_numbers))
    table.add_row("Total Tokens", str(total_tokens))
    table.add_row("Avg Tokens/Line", f"{total_tokens/len(t1_results):.1f}")
    
    console.print("\n")
    console.print(table)


@app.command("run-orchestrator")
def run_orchestrator(
    input_file: str = typer.Option(..., "--in", help="Input file path"),
    output_file: str = typer.Option(..., "--out", help="Output JSONL file"),
):
    """Run unified orchestrator (N0->T1->LLM refine/hints with ops application)."""
    in_path = Path(input_file)
    if not in_path.exists():
        console.print(f"[red]Error: Input file {input_file} not found[/red]")
        raise typer.Exit(1)
    with open(in_path, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                result = orchestrate_block1(line)
                fout.write(json.dumps(result, ensure_ascii=False) + '\n')
            except Exception as e:
                fout.write(json.dumps({'error': str(e), 'raw': line}, ensure_ascii=False) + '\n')
    console.print(f"[green]Saved orchestrator results to {output_file}[/green]")


if __name__ == "__main__":
    app()
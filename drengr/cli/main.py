"""
Elite CLI interface for drengr with rich progress bars and comprehensive options.

This module provides a complete command-line interface using Typer with
rich formatting, progress tracking, and all generation options.
"""

import typer
import time
from typing import Optional
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel

from ..core.generator import generate_dataset_with_new_api
from ..core.exceptions import DrengrError, ConfigurationError, BackendError, ValidationError
from ..services.service_factory import get_service_factory

app = typer.Typer(
    name="drengr",
    help="Elite prompt dataset generation with SOTA defaults",
    add_completion=False
)

console = Console()


@app.command()
def generate(
    total: int = typer.Argument(..., help="Number of prompts to generate"),
    output_path: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path"),
    profile: str = typer.Option("sota", "--profile", "-p", help="Generation profile (sota|fast|cheap|dev)"),
    seed: Optional[int] = typer.Option(None, "--seed", "-s", help="Random seed for reproducibility"),
    embedding_backend: str = typer.Option("auto", "--embedding-backend", help="Embedding service (auto|local|openai|ensemble)"),
    embedding_model: Optional[str] = typer.Option(None, "--embedding-model", help="Specific embedding model"),
    llm_backend: str = typer.Option("auto", "--llm-backend", help="LLM service (auto|local|openai|none)"),
    use_llm_for_paraphrase: bool = typer.Option(True, "--llm-paraphrase/--no-llm-paraphrase", help="Use LLM for paraphrase generation"),
    include_golden: bool = typer.Option(True, "--golden/--no-golden", help="Generate golden responses"),
    preview: int = typer.Option(0, "--preview", help="Number of sample prompts to preview"),
    stream: bool = typer.Option(False, "--stream", help="Stream generation progress"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing output file"),
    run_ablation: bool = typer.Option(False, "--ablation", help="Run ablation experiments"),
    force: bool = typer.Option(False, "--force", help="Bypass validation failures"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
) -> None:
    """Generate prompt dataset with elite SOTA defaults."""
    
    start_time = time.time()
    
    # Display generation info
    console.print(Panel.fit(
        f"[bold blue]drengr[/bold blue] - Elite Prompt Dataset Generation\\n"
        f"Total prompts: [bold]{total:,}[/bold]\\n"
        f"Profile: [bold]{profile}[/bold]\\n"
        f"Seed: [bold]{seed or 'auto'}[/bold]",
        title="Generation Configuration"
    ))
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
            transient=False
        ) as progress:
            
            # Main generation task
            task = progress.add_task("Generating dataset...", total=100)
            
            # Simulate progress updates (in real implementation, this would be integrated)
            progress.update(task, advance=10, description="Initializing configuration...")
            
            # Call the main generate function
            result_path = generate_dataset_with_new_api(
                total=total,
                output_path=output_path,
                profile=profile,
                seed=seed,
                embedding_backend=embedding_backend,
                embedding_model=embedding_model,
                use_llm_for_paraphrase=use_llm_for_paraphrase,
                llm_backend=llm_backend,
                include_golden=include_golden,
                preview=preview,
                stream=stream,
                overwrite=overwrite,
                run_ablation=run_ablation,
                force=force
            )
            
            progress.update(task, completed=100, description="Generation complete!")
        
        # Display results
        generation_time = time.time() - start_time
        
        # Create summary table
        table = Table(title="Generation Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Prompts", f"{total:,}")
        table.add_row("Profile", profile)
        table.add_row("Output File", str(result_path))
        table.add_row("Generation Time", f"{generation_time:.1f}s")
        table.add_row("Prompts/Second", f"{total/generation_time:.1f}")
        
        console.print(table)
        
        # Success message
        console.print(f"\\n[bold green]✓[/bold green] Dataset generated successfully!")
        console.print(f"[dim]Output saved to: {result_path}[/dim]")
        
        if generation_time > 120:
            console.print("[yellow]⚠[/yellow] Generation took longer than 120s target")
        
    except DrengrError as e:
        console.print(f"[bold red]✗[/bold red] Generation failed: {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Unexpected error: {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def demo(
    output_path: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path")
) -> None:
    """Quick demo with 50 prompts using local-only defaults."""
    
    console.print("[bold blue]drengr demo[/bold blue] - Quick smoke test with 50 prompts")
    
    try:
        result_path = generate_dataset_with_new_api(
            total=50,
            output_path=output_path or "./drengr_demo.json",
            profile="dev",
            embedding_backend="mock",
            llm_backend="none",
            preview=5,
            overwrite=True,
            force=True
        )
        
        console.print(f"[bold green]✓[/bold green] Demo complete! Output: {result_path}")
        
    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Demo failed: {e}")
        raise typer.Exit(1)


@app.command()
def profiles() -> None:
    """List available generation profiles."""
    
    profiles_info = {
        "sota": "State-of-the-art profile with industry-standard distributions",
        "fast": "Fast generation profile optimized for speed", 
        "cheap": "Cost-optimized profile minimizing API usage",
        "dev": "Development profile with minimal resource usage"
    }
    
    table = Table(title="Available Profiles")
    table.add_column("Profile", style="cyan")
    table.add_column("Description", style="white")
    
    for profile, description in profiles_info.items():
        table.add_row(profile, description)
    
    console.print(table)



@app.command()
def info() -> None:
    """Display system information and available backends."""
    
    # System info
    import platform
    import sys
    
    table = Table(title="System Information")
    table.add_column("Component", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Platform", platform.system())
    table.add_row("Python Version", sys.version.split()[0])
    table.add_row("drengr Version", "1.0.0")
    
    console.print(table)
    
    # Detect available backends
    try:
        factory = get_service_factory()
        available_backends = factory.detect_available_backends()
        
        backends_table = Table(title="Available Backends")
        backends_table.add_column("Service", style="cyan")
        backends_table.add_column("Backends", style="white")
        
        for service, backends in available_backends.items():
            backends_str = ", ".join(backends) if backends else "none"
            backends_table.add_row(service.title(), backends_str)
        
        console.print(backends_table)
        
        # Service recommendations
        recommendations = factory.get_service_recommendations()
        if recommendations:
            console.print("\n[bold]Recommended Backends:[/bold]")
            for service, backend in recommendations.items():
                console.print(f"• {service.title()}: [green]{backend}[/green]")
    
    except Exception as e:
        console.print(f"\n[yellow]⚠[/yellow] Could not detect backends: {e}")
        console.print("\n[bold]Default Backends:[/bold]")
        console.print("• Embedding: auto, mock, local, openai")
        console.print("• LLM: auto, mock, local, openai, none")


@app.command()
def benchmark(
    size: int = typer.Option(100, "--size", "-s", help="Dataset size for benchmark"),
    profile: str = typer.Option("fast", "--profile", "-p", help="Profile to benchmark"),
    iterations: int = typer.Option(3, "--iterations", "-i", help="Number of benchmark iterations")
) -> None:
    """Run performance benchmarks."""
    
    console.print(f"[blue]Running benchmark:[/blue] {size} prompts, {iterations} iterations")
    
    import tempfile
    import statistics
    
    times = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        
        task = progress.add_task("Running benchmarks...", total=iterations)
        
        for i in range(iterations):
            with tempfile.NamedTemporaryFile(suffix=".json", delete=True) as tmp_file:
                start_time = time.time()
                
                try:
                    generate_dataset_with_new_api(
                        total=size,
                        output_path=tmp_file.name,
                        profile=profile,
                        embedding_backend="mock",
                        llm_backend="mock",
                        force=True,
                        overwrite=True
                    )
                    
                    execution_time = time.time() - start_time
                    times.append(execution_time)
                    
                except Exception as e:
                    console.print(f"[red]✗[/red] Benchmark iteration {i+1} failed: {e}")
                    continue
                
                progress.update(task, advance=1, description=f"Iteration {i+1}/{iterations}")
    
    if times:
        # Calculate statistics
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0
        
        # Display results
        results_table = Table(title="Benchmark Results")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="white")
        
        results_table.add_row("Dataset Size", f"{size:,} prompts")
        results_table.add_row("Profile", profile)
        results_table.add_row("Iterations", str(iterations))
        results_table.add_row("Average Time", f"{avg_time:.2f}s")
        results_table.add_row("Min Time", f"{min_time:.2f}s")
        results_table.add_row("Max Time", f"{max_time:.2f}s")
        results_table.add_row("Std Deviation", f"{std_dev:.2f}s")
        results_table.add_row("Avg Throughput", f"{size/avg_time:.1f} prompts/sec")
        
        console.print(results_table)
        
        # Performance assessment
        if avg_time < 30:
            console.print("[green]✓[/green] Excellent performance!")
        elif avg_time < 60:
            console.print("[yellow]⚠[/yellow] Good performance")
        else:
            console.print("[red]⚠[/red] Performance could be improved")
    
    else:
        console.print("[red]✗[/red] All benchmark iterations failed")
        raise typer.Exit(1)


@app.command()
def test_backends() -> None:
    """Test available backends and their health."""
    
    console.print("[blue]Testing backend health...[/blue]")
    
    try:
        factory = get_service_factory()
        available_backends = factory.detect_available_backends()
        
        # Test each backend combination
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            embedding_backends = available_backends.get("embedding", ["mock"])
            llm_backends = available_backends.get("llm", ["mock", "none"])
            
            total_tests = len(embedding_backends) * len(llm_backends)
            task = progress.add_task("Testing backends...", total=total_tests)
            
            for emb_backend in embedding_backends:
                for llm_backend in llm_backends:
                    progress.update(task, description=f"Testing {emb_backend}/{llm_backend}")
                    
                    try:
                        container = factory.create_service_container(
                            embedding_backend=emb_backend,
                            llm_backend=llm_backend
                        )
                        
                        health_status = factory.validate_service_health(container)
                        
                        all_healthy = all(health_status.values())
                        results.append({
                            "embedding": emb_backend,
                            "llm": llm_backend,
                            "healthy": all_healthy,
                            "details": health_status
                        })
                        
                    except Exception as e:
                        results.append({
                            "embedding": emb_backend,
                            "llm": llm_backend,
                            "healthy": False,
                            "error": str(e)
                        })
                    
                    progress.update(task, advance=1)
        
        # Display results
        results_table = Table(title="Backend Health Test Results")
        results_table.add_column("Embedding", style="cyan")
        results_table.add_column("LLM", style="cyan")
        results_table.add_column("Status", style="white")
        results_table.add_column("Details", style="dim")
        
        for result in results:
            status = "[green]✓ Healthy[/green]" if result["healthy"] else "[red]✗ Unhealthy[/red]"
            
            if result["healthy"]:
                details = "All services operational"
            elif "error" in result:
                details = result["error"][:50] + "..." if len(result["error"]) > 50 else result["error"]
            else:
                unhealthy_services = [k for k, v in result.get("details", {}).items() if not v]
                details = f"Issues: {', '.join(unhealthy_services)}" if unhealthy_services else "Unknown issue"
            
            results_table.add_row(
                result["embedding"],
                result["llm"],
                status,
                details
            )
        
        console.print(results_table)
        
        # Summary
        healthy_count = sum(1 for r in results if r["healthy"])
        total_count = len(results)
        
        console.print(f"\n[bold]Summary:[/bold] {healthy_count}/{total_count} backend combinations healthy")
        
        if healthy_count == 0:
            console.print("[red]⚠[/red] No healthy backend combinations found!")
            raise typer.Exit(1)
        elif healthy_count < total_count:
            console.print("[yellow]⚠[/yellow] Some backend combinations have issues")
        else:
            console.print("[green]✓[/green] All backend combinations are healthy")
    
    except Exception as e:
        console.print(f"[red]✗[/red] Backend testing failed: {e}")
        raise typer.Exit(1)


def main():
    app()

if __name__ == "__main__":
    main()
"""Main entry point for Voix du Client application."""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
from loguru import logger

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from voix_du_client.analyzer import FeedbackAnalyzer
from voix_du_client.config import Config
from voix_du_client.models import AnalysisConfig, FeedbackItem


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logger.remove()  # Remove default handler
    
    # Console handler
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # File handler
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logger.add(
        log_dir / "voix_du_client.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation="10 MB",
        retention="30 days",
        compression="zip"
    )


def load_feedback_from_csv(filepath: Path) -> List[FeedbackItem]:
    """Load feedback items from CSV file."""
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} rows from {filepath}")
        
        if 'text' not in df.columns:
            raise ValueError("CSV file must contain a 'text' column")
        
        feedback_items = []
        for idx, row in df.iterrows():
            item_id = str(row.get('id', idx))
            text = str(row['text']).strip()
            
            if text and text.lower() != 'nan':
                item = FeedbackItem(id=item_id, text=text)
                feedback_items.append(item)
        
        logger.info(f"Created {len(feedback_items)} valid feedback items")
        return feedback_items
        
    except Exception as e:
        logger.error(f"Error loading CSV file {filepath}: {e}")
        raise


def run_analysis(csv_path: str, k: int, max_features: int, 
                ngram_min: int, ngram_max: int, random_state: int,
                output_path: Optional[str] = None, save_model: bool = False) -> None:
    """Run feedback analysis from command line."""
    
    # Load configuration
    config = Config()
    
    # Load data
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    feedback_items = load_feedback_from_csv(csv_file)
    
    if len(feedback_items) < k:
        raise ValueError(f"Not enough feedback items ({len(feedback_items)}) for {k} clusters")
    
    # Create analysis configuration
    analysis_config = AnalysisConfig(
        k=k,
        max_features=max_features,
        ngram_range=(ngram_min, ngram_max),
        random_state=random_state,
        min_text_length=config.min_text_length
    )
    
    # Run analysis
    analyzer = FeedbackAnalyzer(config)
    logger.info(f"Starting analysis with {len(feedback_items)} items, k={k}")
    
    result = analyzer.analyze(feedback_items, analysis_config)
    
    # Display results
    print("\n" + "="*80)
    print("RÉSULTATS DE L'ANALYSE VOIX DU CLIENT")
    print("="*80)
    print(f"Nombre total de verbatims: {result.total_items}")
    print(f"Nombre de clusters: {result.num_clusters}")
    print(f"Score silhouette: {result.silhouette_score:.3f}")
    print(f"Temps de traitement: {result.processing_time:.2f}s")
    print("\n")
    
    # Display cluster details
    for i, cluster in enumerate(result.clusters, 1):
        print(f"CLUSTER {cluster.id} ({cluster.size} verbatims - {cluster.percentage:.1f}%)")
        print("-" * 60)
        print("Termes principaux:")
        for j, term in enumerate(cluster.top_terms[:8], 1):
            print(f"  {j}. {term}")
        
        if cluster.representative_texts:
            print("\nExemples de verbatims:")
            for j, text in enumerate(cluster.representative_texts[:2], 1):
                print(f"  {j}. {text}")
        print("\n")
    
    # Save results if requested
    if output_path:
        output_file = Path(output_path)
        
        # Create detailed results DataFrame
        results_data = []
        for item in feedback_items:
            cluster_info = next((c for c in result.clusters if c.id == item.cluster), None)
            results_data.append({
                'id': item.id,
                'text': item.text,
                'clean_text': item.clean_text,
                'cluster': item.cluster,
                'cluster_size': cluster_info.size if cluster_info else 0,
                'cluster_percentage': cluster_info.percentage if cluster_info else 0,
                'top_terms': ', '.join(cluster_info.top_terms[:5]) if cluster_info else ''
            })
        
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"Results saved to {output_file}")
        print(f"Résultats sauvegardés dans: {output_file}")
    
    # Save model if requested
    if save_model:
        model_path = config.models_dir / f"model_k{k}_{int(result.timestamp.timestamp())}.joblib"
        analyzer.save_model(model_path)
        print(f"Modèle sauvegardé dans: {model_path}")


def run_dashboard():
    """Run the Streamlit dashboard."""
    try:
        import streamlit.web.cli as stcli
        from voix_du_client.dashboard import main
        
        # Run dashboard
        logger.info("Starting Streamlit dashboard")
        main()
        
    except ImportError:
        logger.error("Streamlit not available. Install with: pip install streamlit")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Voix du Client - Analyse des irritants clients",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  # Lancer le dashboard web
  voix-du-client dashboard
  
  # Analyse en ligne de commande
  voix-du-client analyze --csv feedback.csv --k 5
  
  # Analyse avec sauvegarde
  voix-du-client analyze --csv feedback.csv --k 5 --output results.csv --save-model
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commandes disponibles')
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Lancer le dashboard web')
    
    # Analysis command
    analyze_parser = subparsers.add_parser('analyze', help='Analyse en ligne de commande')
    analyze_parser.add_argument('--csv', type=str, required=True, 
                               help='Chemin vers le fichier CSV de verbatims')
    analyze_parser.add_argument('--k', type=int, default=5,
                               help='Nombre de clusters (défaut: 5)')
    analyze_parser.add_argument('--max-features', type=int, default=5000,
                               help='Taille du vocabulaire TF-IDF (défaut: 5000)')
    analyze_parser.add_argument('--ngram-min', type=int, default=1,
                               help='N-gram minimum (défaut: 1)')
    analyze_parser.add_argument('--ngram-max', type=int, default=2,
                               help='N-gram maximum (défaut: 2)')
    analyze_parser.add_argument('--random-state', type=int, default=42,
                               help='Graine aléatoire (défaut: 42)')
    analyze_parser.add_argument('--output', type=str,
                               help='Fichier de sortie pour les résultats (CSV)')
    analyze_parser.add_argument('--save-model', action='store_true',
                               help='Sauvegarder le modèle entraîné')
    analyze_parser.add_argument('--log-level', type=str, default='INFO',
                               choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                               help='Niveau de log (défaut: INFO)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Setup logging
    log_level = getattr(args, 'log_level', 'INFO')
    setup_logging(log_level)
    
    try:
        if args.command == 'dashboard':
            run_dashboard()
        
        elif args.command == 'analyze':
            run_analysis(
                csv_path=args.csv,
                k=args.k,
                max_features=args.max_features,
                ngram_min=args.ngram_min,
                ngram_max=args.ngram_max,
                random_state=args.random_state,
                output_path=args.output,
                save_model=args.save_model
            )
    
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
double_perovskite_full_scraper.py

Robust, resumable scraper for double perovskites on Materials Project.

Features:
- Multiple search strategies:
    * formula_search (oxide/halide/chalcogenide patterns)
    * spacegroup + nsites combinations
    * crystal_system + nelements sweep
    * text/tag matching ("perovskite" in metadata)
- Pagination with retries and rate limiting
- Parallelized requests for filter combinations
- Checkpointing: writes CSV and a seen_ids JSON to resume safely
- Optional: fetch full structure for stronger "Perovskite" mineral check (requires pymatgen)
- Usage: python double_perovskite_full_scraper.py --api-key YOUR_KEY
"""

import argparse
import concurrent.futures
import json
import logging
import math
import os
import sys
import threading
import time
from collections import OrderedDict
from typing import Dict, List, Optional

import pandas as pd
import requests

# ----------------------------
# Logging config
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DP-Scraper")

# ----------------------------
# Config / Constants
# ----------------------------
MP_BASE = "https://api.materialsproject.org"
SUMMARY_ENDPOINT = "/materials/summary/"
PER_PAGE = 250
MAX_RETRIES = 3
RETRY_BACKOFF = 3  # seconds * attempt multiplier
RATE_LIMIT_DELAY = 0.12  # seconds between API calls (polite)

DEFAULT_OUTPUT = "double_perovskites_results.csv"
SEEN_IDS_FILE = "double_perovskites_seen.json"

# More restrictive space groups for true perovskites
DEFAULT_SPACE_GROUPS = [
    # Common perovskite space groups
    "Pm-3m", "Pnma", "R-3c", "I4/mcm", "P4/mbm", "Im-3",
    "Fm-3m", "I4/m", "I4/mmm", "P4/mnc", "R-3m", "Cmcm"
]

# More restrictive site counts for conventional perovskite cells
DEFAULT_SITE_COUNTS = [5, 8, 10, 12, 15, 20, 24, 30, 40, 60, 80]

CRYSTAL_SYSTEMS = ["cubic", "tetragonal", "orthorhombic", "monoclinic", "trigonal", "hexagonal", "rhombohedral", "triclinic"]

# Expanded formula patterns to include chalcogenides
FORMULA_PATTERNS = [
    # oxide patterns - use wildcards for actual elements
    "*O6", "*O3", "*O12",
    # halide patterns
    "*F6", "*Cl6", "*Br6", "*I6",
    # chalcogenide patterns
    "*S3", "*Se3", "*Te3",
    "*S6", "*Se6", "*Te6"
]



# Expanded anion set to include chalcogenides
ANIONS = {"O", "F", "Cl", "Br", "I", "N", "S", "Se", "Te"}

# Common perovskite A-site cations
COMMON_A_SITES = {"Li", "Na", "K", "Rb", "Cs", "Mg", "Ca", "Sr", "Ba", "La", "Y"}
# Common perovskite B-site cations  
COMMON_B_SITES = {"Ti", "Zr", "Hf", "V", "Nb", "Ta", "Cr", "Mo", "W", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Al", "Ga", "In", "Sn", "Pb", "Bi"}

# Thread-safety
seen_lock = threading.Lock()
write_lock = threading.Lock()

# Try to import optional pymatgen for structure validation
try:
    from pymatgen.ext.matproj import MPRester
    from pymatgen.analysis.structure_matcher import StructureMatcher
    HAVE_PMG = True
except Exception:
    HAVE_PMG = False


# ----------------------------
# Utilities and client
# ----------------------------
class MPClient:
    def __init__(self, api_key: str, rate_delay: float = RATE_LIMIT_DELAY):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"X-API-KEY": self.api_key})
        self.rate_delay = rate_delay

    def get(self, url: str, params: Dict) -> Optional[Dict]:
        # Basic retry/backoff
        attempt = 0
        while attempt < MAX_RETRIES:
            attempt += 1
            try:
                time.sleep(self.rate_delay)
                r = self.session.get(url, params=params, timeout=30)
                r.raise_for_status()
                return r.json()
            except requests.RequestException as exc:
                logger.warning(f"Request error (attempt {attempt}) for {url} with params {params}: {exc}")
                time.sleep(RETRY_BACKOFF * attempt)
        logger.error(f"Failed request after {MAX_RETRIES} attempts: {url} {params}")
        return None


# ----------------------------
# Scraper class
# ----------------------------
class DoublePerovskiteFullScraper:
    def __init__(self, api_key: str, output_csv: str = DEFAULT_OUTPUT,
                 seen_ids_file: str = SEEN_IDS_FILE, fetch_structures: bool = False,
                 max_workers: int = 6):
        self.client = MPClient(api_key)
        self.output_csv = output_csv
        self.seen_ids_file = seen_ids_file
        self.fetch_structures = fetch_structures
        self.max_workers = max_workers

        # results stored as ordered dict keyed by material_id for deterministic output
        self.results: Dict[str, Dict] = OrderedDict()
        self.seen_ids = set()
        if os.path.exists(self.seen_ids_file):
            try:
                with open(self.seen_ids_file, "r") as fh:
                    self.seen_ids = set(json.load(fh))
                logger.info(f"Loaded {len(self.seen_ids)} seen IDs from {self.seen_ids_file}")
            except Exception:
                logger.warning("Could not load seen ids file; starting fresh.")

        # optional pymatgen client for structure checks
        self.pmg_rester = None
        if self.fetch_structures and HAVE_PMG:
            self.pmg_rester = MPRester(api_key)
            logger.info("pymatgen MPRester initialized for structure-level checks")
        elif self.fetch_structures:
            logger.warning("fetch_structures requested but pymatgen not available. Install pymatgen to enable.")

    # ----------------------------
    # Core API helpers
    # ----------------------------
    def _summary_query(self, params: Dict) -> Optional[Dict]:
        url = MP_BASE + SUMMARY_ENDPOINT
        return self.client.get(url, params)

    def _paginate_query(self, base_params: Dict) -> List[Dict]:
        """Return list of all material summary objects matching params (handles pagination)."""
        page = 1
        all_data = []
        while True:
            params = dict(base_params)
            params["_page"] = page
            params["_per_page"] = PER_PAGE
            resp = self._summary_query(params)
            if not resp:
                break
            data = resp.get("data", [])
            if not data:
                break
            all_data.extend(data)
            meta_total = resp.get("meta", {}).get("total_doc")
            # if metadata present and we've retrieved all
            if meta_total is not None and len(all_data) >= meta_total:
                break
            # otherwise step page counter
            page += 1
            # safety: prevent infinite loop
            if page > 2000:
                logger.error("Too many pages, breaking to avoid infinite loop.")
                break
        return all_data

    # ----------------------------
    # Improved Heuristics / Validation
    # ----------------------------
    def is_likely_double_perovskite(self, material: Dict) -> bool:
        """Strict heuristic-based check for double perovskites."""
        try:
            mid = material.get("material_id", "<no-id>")
            formula = material.get("formula_pretty", "") or material.get("formula", "")
            anon = material.get("formula_anonymous", "") or ""
            elements = set(material.get("elements", []))
            nelements = int(material.get("nelements", 0) or 0)
            nsites = int(material.get("nsites", 0) or 0)

            # Stricter bounds for perovskites
            if nelements < 3 or nelements > 6:  # Reduced from 2-8
                return False
            if nsites < 5 or nsites > 120:  # Reduced from 4-200
                return False

            # Quick accept if MP tags or fields include "perovskite"
            text_blob = json.dumps(material).lower()
            if "perovskite" in text_blob:
                logger.debug(f"Quick accept {mid}: perovskite in metadata")
                return True

            # Check for common perovskite anions
            anions_present = elements & ANIONS
            if not anions_present:
                return False

            # Check for common perovskite cations
            cations = elements - anions_present
            common_cations = cations & (COMMON_A_SITES | COMMON_B_SITES)
            if len(common_cations) < 2:  # Need at least 2 common cations
                return False

            # NEW: Check for A2BB'O6 type stoichiometry in elements
            if nelements >= 4:
                # Count occurrences of common perovskite elements
                common_perovskite_elements = len(elements & (COMMON_A_SITES | COMMON_B_SITES | ANIONS))
                if common_perovskite_elements >= 4:  # At least A, B, B', X
                    logger.debug(f"Stoichiometry match {mid}: {common_perovskite_elements} common perovskite elements")
                    return True

            # Strict formula pattern matching
            formula_clean = formula.replace(" ", "")
            for patt in ["O6", "S6", "Se6", "Te6", "F6", "Cl6", "Br6", "I6"]:
                if patt in formula_clean:
                    # Additional check: should have multiple cations
                    if len(cations) >= 2:
                        logger.debug(f"Pattern match {mid}: {patt} in formula with {len(cations)} cations")
                        return True

            # Check anonymous formula for A2BB'X6 pattern
            anon_lower = anon.lower()
            if ("a2" in anon_lower or "aa'" in anon_lower) and ("x6" in anon_lower or "o6" in anon_lower):
                logger.debug(f"Anonymous formula match {mid}: {anon}")
                return True

            # Check for perovskite-like stoichiometry in elements
            if self._has_perovskite_stoichiometry(material):
                logger.debug(f"Stoichiometry match {mid}")
                return True

            return False
        except Exception as e:
            logger.warning(f"Error in perovskite heuristic for material {mid}: {e}")
            return False

    def _has_perovskite_stoichiometry(self, material: Dict) -> bool:
        """Check if material has ABX3 or A2BB'X6 stoichiometry."""
        try:
            formula = material.get("formula_pretty", "") or material.get("formula", "")
            elements = material.get("elements", [])
            nelements = len(elements)
            
            if nelements == 3:  # ABX3
                return True
            elif nelements >= 4:  # A2BB'X6 or variants
                # Check if formula suggests multiple cations with one anion type
                formula_lower = formula.lower()
                anion_counts = {}
                for anion in ANIONS:
                    if anion.lower() in formula_lower:
                        # Simple check for common perovskite ratios
                        if '3' in formula.split(anion)[-1] or '6' in formula.split(anion)[-1]:
                            return True
            return False
        except Exception:
            return False

    # Optional stronger validation by fetching structure
    def structure_level_check(self, material_id: str) -> bool:
        if not self.pmg_rester:
            return False
        try:
            struct = self.pmg_rester.get_structure_by_material_id(material_id)
            # Basic structure checks for perovskites
            if len(struct) < 5 or len(struct) > 40:  # Reasonable perovskite sizes
                return False
                
            # Check for approximate cubic/tetragonal/orthorhombic symmetry
            lattice = struct.lattice
            lengths = lattice.abc
            angles = lattice.angles
            
            # Allow some distortion but not too extreme
            max_length_ratio = max(lengths) / min(lengths)
            if max_length_ratio > 3.0:  # Too distorted for perovskite
                return False
                
            return True
        except Exception as e:
            logger.debug(f"Structure-level check failed for {material_id}: {e}")
            return False
        
    # ----------------------------
    # NEW: Load specific MP IDs from CSV
    # ----------------------------
    def strategy_csv_mpids(self, csv_path: str, id_column: str = None):
        """
        Load a CSV containing MP IDs, fetch their MP summary entries,
        and process them through the existing pipeline.

        Parameters
        ----------
        csv_path : str
            Path to CSV file.
        id_column : str or None
            Column containing the mp-id. If None, the first column is assumed.
        """
        logger.info(f"Reading MP IDs from CSV: {csv_path}")

        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            logger.error(f"Could not read CSV {csv_path}: {exc}")
            return

        # Determine which column contains the mp-id strings
        if id_column is None:
            id_column = df.columns[0]  # your example CSV uses mp-id in first col

        if id_column not in df.columns:
            logger.error(f"Column '{id_column}' not found in CSV.")
            return

        mp_ids = df[id_column].dropna().astype(str).unique().tolist()
        logger.info(f"Found {len(mp_ids)} MP IDs in CSV")

        # Query each mp-id
        for mpid in mp_ids:
            if mpid in self.seen_ids:
                logger.debug(f"{mpid} already processed, skipping.")
                continue

            params = {
                "material_ids": mpid,
                "_all_fields": True
            }

            data = self._paginate_query(params)
            if not data:
                logger.warning(f"No data returned for {mpid}")
                continue

            # Should be one entry, but loop anyway
            for material in data:
                self._consider_material(material)

            self._save_checkpoint()

        logger.info(f"Finished CSV ID scraping for {len(mp_ids)} IDs.")

    # ----------------------------
    # Extraction / saving
    # ----------------------------
    def _extract_record(self, material: Dict) -> Dict:
        return {
            "material_id": material.get("material_id", ""),
            "formula": material.get("formula_pretty", material.get("formula", "")),
            "formula_anonymous": material.get("formula_anonymous", ""),
            "spacegroup": material.get("spacegroup_symbol", ""),
            "crystal_system": material.get("crystal_system", ""),
            "nsites": material.get("nsites", ""),
            "nelements": material.get("nelements", ""),
            "density": material.get("density", ""),
            "volume": material.get("volume", ""),
            "formation_energy_per_atom": material.get("formation_energy_per_atom", ""),
            "energy_above_hull": material.get("energy_above_hull", ""),
            "band_gap": material.get("band_gap", ""),
            "is_metal": material.get("is_metal", ""),
            "total_magnetization": material.get("total_magnetization", ""),
            "is_theoretical": material.get("theoretical", ""),
            "elements": ", ".join(material.get("elements", [])),
            "tags": ", ".join(material.get("tags", []) if isinstance(material.get("tags", []), list) else [])
        }

    def _save_checkpoint(self):
        # write CSV and seen ids
        with write_lock:
            if self.results:
                df = pd.DataFrame(list(self.results.values()))
                df.to_csv(self.output_csv, index=False)
                logger.info(f"Saved {len(self.results)} results to {self.output_csv}")
            # persist seen ids
            try:
                with open(self.seen_ids_file, "w") as fh:
                    json.dump(list(self.seen_ids), fh)
                logger.debug(f"Saved {len(self.seen_ids)} seen ids to {self.seen_ids_file}")
            except Exception as exc:
                logger.warning(f"Could not save seen ids: {exc}")

    # ----------------------------
    # Search Strategies
    # ----------------------------
    def strategy_formula_patterns(self):
        """Use formula_search-like queries to find obvious patterns."""
        logger.info("Starting strategy: formula pattern searches...")
        
        # Search wildcard patterns
        for patt in FORMULA_PATTERNS:
            params = {
                "formula": patt,
                "_all_fields": True
            }
            items = self._paginate_query(params)
            logger.info(f"Pattern {patt} returned {len(items)} candidates")
            for m in items:
                self._consider_material(m)
            self._save_checkpoint()
        

    def strategy_spacegroup_site(self, spacegroups: List[str], site_counts: List[int]):
        """Search by spacegroup and site counts. Parallelizes over combinations."""
        logger.info("Starting strategy: spacegroup + nsites sweep (parallelized)")
        combos = []
        for sg in spacegroups:
            for n in site_counts:
                combos.append((sg, n))

        def worker(combo):
            sg, nsites = combo
            params = {
                "spacegroup_symbol": sg,
                "nsites": nsites,
                "nelements_min": 3, 
                "nelements_max": 6,  # Stricter range for perovskites
                "_all_fields": True
            }
            items = self._paginate_query(params)
            logger.info(f"{sg} | {nsites} -> {len(items)} items")
            for m in items:
                self._consider_material(m)
            # local checkpoint
            self._save_checkpoint()

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = list(ex.map(worker, combos))

    def strategy_crystal_system_nelements(self):
        """Search by crystal_system and a sweep of nelements (3..6)."""
        logger.info("Starting strategy: crystal system + nelements sweep")
        for system in CRYSTAL_SYSTEMS:
            for ne in range(3, 7):
                params = {
                    "crystal_system": system,
                    "nelements": ne,
                    "_all_fields": True
                }
                items = self._paginate_query(params)
                logger.info(f"System={system} nelements={ne} -> {len(items)} items")
                for m in items:
                    self._consider_material(m)
                self._save_checkpoint()

    def strategy_text_search(self, text_terms: List[str] = ["perovskite", "double perovskite"]):
        """Use targeted text search with perovskite-relevant chemsys."""
        logger.info("Starting strategy: local text-based filtering")

        # Expanded chemsys list to include chalcogenides
        chemsys_list = [
            "O", "O-Ti", "O-Fe", "O-Ba", "O-Sr", "O-Ca", "O-La", 
            "Br-Bi", "Cl-Bi", "Br-Ag", "I-Pb", "Cl-Pb", "I-Cs",
            "S", "S-Sr", "S-Ba", "Se", "Se-Sr", "Se-Ba", "Te", "Te-Sr"
        ]
        for chemsys in chemsys_list:
            params = {
                "chemsys": chemsys,
                "_all_fields": True,
                "_per_page": PER_PAGE,
                "_page": 1
            }
            items = self._paginate_query(params)
            logger.info(f"chemsys {chemsys} -> {len(items)} items")
            for m in items:
                text = json.dumps(m).lower()
                if any(term in text for term in text_terms):
                    self._consider_material(m)
            self._save_checkpoint()

    # ----------------------------
    # Main consideration and dedupe
    # ----------------------------
    def _consider_material(self, material: Dict):
        mid = material.get("material_id")
        if not mid:
            return
        with seen_lock:
            if mid in self.seen_ids:
                return
            # apply heuristic test
            likely = self.is_likely_double_perovskite(material)
            if not likely and self.fetch_structures:
                # try stronger structure-level check if enabled
                if self.structure_level_check(mid):
                    likely = True
            if not likely:
               
                return
            # accept material
            rec = self._extract_record(material)
            self.results[mid] = rec
            self.seen_ids.add(mid)
            logger.info(f"ACCEPTED {mid} | {rec['formula']} | sg={rec['spacegroup']}")

    # ----------------------------
    # Orchestration
    # ----------------------------
    def run_all(self, space_groups: Optional[List[str]] = None, site_counts: Optional[List[int]] = None):
        if space_groups is None:
            space_groups = DEFAULT_SPACE_GROUPS
        if site_counts is None:
            site_counts = DEFAULT_SITE_COUNTS

        try:
            # Strategy 1 — formula patterns
            self.strategy_formula_patterns()

            # Strategy 2 — spacegroup + site counts (parallel)
            self.strategy_spacegroup_site(space_groups, site_counts)

            # Strategy 3 — crystal system + nelements sweep
            #self.strategy_crystal_system_nelements()

            # Strategy 4 — tag/text-based targeted chemsys search
            self.strategy_text_search()

        finally:
            # final save
            self._save_checkpoint()
            logger.info("All strategies completed (or aborted). Final checkpoint saved.")

# ----------------------------
# CLI / main
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Comprehensive double perovskite scraper for Materials Project")
    p.add_argument("--api-key", type=str, default="IDScwWdmFrlYPrCFBI31h23dSNlaIEvE", help="Materials Project API key (or set env MP_API_KEY)")
    p.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="CSV output filename")
    p.add_argument("--seen-file", type=str, default=SEEN_IDS_FILE, help="Seen IDs JSON to allow resume")
    p.add_argument("--fetch-structures", action="store_true", help="If set, fetch full structures for stronger checks (requires pymatgen)")
    p.add_argument("--workers", type=int, default=18, help="Number of threads for parallel queries")
    p.add_argument("--no-formula-strategy", action="store_true", help="Skip formula pattern searches")
    p.add_argument("--csv-ids", type=str, help="CSV file containing MP IDs to scrape")

    return p.parse_args()

def main():
    args = parse_args()
    api_key = args.api_key or os.environ.get("MP_API_KEY")
    if not api_key:
        logger.error("No Materials Project API key provided. Use --api-key or set MP_API_KEY env var.")
        sys.exit(1)

    scraper = DoublePerovskiteFullScraper(
        api_key=api_key,
        output_csv=args.output,
        seen_ids_file=args.seen_file,
        fetch_structures=args.fetch_structures,
        max_workers=args.workers
    )

    logger.info("Starting double perovskite scraping run")
    start = time.time()
    # Option to skip formula strategy if requested
    if args.no_formula_strategy:
        scraper.strategy_spacegroup_site(DEFAULT_SPACE_GROUPS, DEFAULT_SITE_COUNTS)
        #scraper.strategy_crystal_system_nelements()
        scraper.strategy_text_search()
    elif args.csv_ids:
        scraper.strategy_csv_mpids(args.csv_ids)
    else:
        scraper.run_all()
    end = time.time()
    logger.info(f"Scraping finished in {end - start:.1f} seconds. Results saved to {args.output}")

if __name__ == "__main__":
    main()

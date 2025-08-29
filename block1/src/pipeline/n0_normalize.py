"""
N0 - Deterministic text normalization with zero information loss.
"""
import re
import unicodedata
from typing import List, Tuple, Optional
from ..common.schemas import NormalizationResult, Pair


class N0Normalizer:
    """Deterministic normalizer for dental text."""
    
    def __init__(self):
        # Compile regex patterns for efficiency
        self.hebrew_pattern = re.compile(r'[\u0590-\u05FF]+')
        self.english_pattern = re.compile(r'[a-zA-Z]+')
        self.digit_pattern = re.compile(r'\d+')
        
        # Pattern for number pairs: A sep B where sep in {/, \, -}
        self.pair_pattern = re.compile(r'(\d+)\s*([/\\-])\s*(\d+)')
        
        # Unit normalization mappings
        self.unit_mappings = {
            'ממ': 'mm',
            'מ"מ': 'mm',
            'מילימטר': 'mm',
            'millimeter': 'mm',
            'מעלות': '°'
        }
    
    def normalize_n0(self, raw_text: str) -> NormalizationResult:
        """
        Main normalization function.
        
        Args:
            raw_text: Input text to normalize
            
        Returns:
            NormalizationResult with normalized text and metadata
        """
        if not raw_text:
            return NormalizationResult(raw_text="", normalized_text="")

        # Unicode normalization
        text = unicodedata.normalize('NFC', raw_text)
        notes: List[str] = []

        # Detect pairs first (original spans)
        pairs = self._find_pairs(text)

        # Spacing between letters/digits
        text, space_notes = self._insert_spaces(text)
        notes.extend(space_notes)

        # Units normalization
        text, units = self._normalize_units(text)

    # Removed glued Hebrew splitting (handled later or via variants) to avoid false splits

        # Update pair spans after modifications
        pairs = self._update_pair_spans(text, pairs)

        # Numbers/dates/times
        numbers = self._extract_numbers(text)
        dates = self._find_dates(text)
        times = self._find_times(text)

        # Trim trailing whitespace to stabilize downstream spans
        text = text.rstrip()
        return NormalizationResult(
            raw_text=raw_text,
            normalized_text=text,
            numbers=numbers,
            pairs=pairs,
            units_found=units,
            dates=dates,
            times=times,
            notes=notes
        )
    
    def _find_pairs(self, text: str) -> List[Pair]:
        """Find number pairs in text."""
        pairs = []
        for match in self.pair_pattern.finditer(text):
            pairs.append(Pair(
                text=match.group(0),
                A=match.group(1),
                B=match.group(3),
                sep=match.group(2),
                span=[match.start(), match.end()]
            ))
        return pairs
    
    def _insert_spaces(self, text: str) -> Tuple[str, List[str]]:
        """Insert spaces between letters and digits."""
        notes = []
        
        # Hebrew letter followed by digit
        new_text = re.sub(r'([\u0590-\u05FF])(\d)', r'\1 \2', text)
        if new_text != text:
            notes.append("inserted_space_hebrew_digit")
            text = new_text
        
        # Digit followed by Hebrew letter
        new_text = re.sub(r'(\d)([\u0590-\u05FF])', r'\1 \2', text)
        if new_text != text:
            notes.append("inserted_space_digit_hebrew")
            text = new_text
        
        # English letter followed by digit (but preserve existing patterns like multi-unit)
        new_text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
        if new_text != text:
            notes.append("inserted_space_english_digit")
            text = new_text
            
        # Digit followed by English letter
        new_text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
        if new_text != text:
            notes.append("inserted_space_digit_english")
            text = new_text
        
        return text, notes
    
    def _normalize_units(self, text: str) -> Tuple[str, List[str]]:
        """Normalize existing units in text."""
        units_found = []
        
        for original, normalized in self.unit_mappings.items():
            if original in text:
                text = text.replace(original, normalized)
                if normalized not in units_found:
                    units_found.append(normalized)
        
        # Check for degree symbol
        if '°' in text or 'מעלות' in text:
            if '°' not in units_found:
                units_found.append('°')
        
        return text, units_found
    
    def _update_pair_spans(self, text: str, original_pairs: List[Pair]) -> List[Pair]:
        """Update pair spans after text modifications."""
        updated_pairs = []
        
        for pair in original_pairs:
            # Re-find the pair in modified text
            pattern = re.escape(pair.A) + r'\s*' + re.escape(pair.sep) + r'\s*' + re.escape(pair.B)
            match = re.search(pattern, text)
            
            if match:
                updated_pairs.append(Pair(
                    text=match.group(0),
                    A=pair.A,
                    B=pair.B,
                    sep=pair.sep,
                    span=[match.start(), match.end()]
                ))
        
        return updated_pairs
    
    def _extract_numbers(self, text: str) -> List[str]:
        """Extract all numbers from text."""
        numbers = []
        
        # Find all digit sequences
        for match in self.digit_pattern.finditer(text):
            numbers.append(match.group(0))
        
        return numbers
    
    def _find_dates(self, text: str) -> List[str]:
        """Find date patterns in text."""
        dates = []
        
        # Common date patterns
        date_patterns = [
            r'\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}',  # DD/MM/YYYY, DD-MM-YYYY, etc.
            r'\d{2,4}[/.-]\d{1,2}[/.-]\d{1,2}',  # YYYY/MM/DD
        ]
        
        for pattern in date_patterns:
            for match in re.finditer(pattern, text):
                dates.append(match.group(0))
        
        return dates
    
    def _find_times(self, text: str) -> List[str]:
        """Find time patterns in text."""
        times = []
        
        # Time patterns
        time_patterns = [
            r'\d{1,2}:\d{2}(?::\d{2})?',  # HH:MM or HH:MM:SS
            r'\d{1,2}[:.]\d{2}\s*(?:AM|PM|am|pm)',  # 12-hour format
        ]
        
        for pattern in time_patterns:
            for match in re.finditer(pattern, text):
                times.append(match.group(0))
        
        return times

    # _split_glued_hebrew removed intentionally


# Convenience function for direct use
def normalize_n0(raw_text: str) -> NormalizationResult:
    """
    Normalize text using N0 rules.
    
    Args:
        raw_text: Input text
        
    Returns:
        NormalizationResult
    """
    normalizer = N0Normalizer()
    return normalizer.normalize_n0(raw_text)
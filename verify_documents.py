#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TC Kimlik Kartı ve Başvuru Formu Doğrulama Sistemi (PoC)

Author: Mustafa Kursad Baser
Version: 1.0.0

Özellikler:
- Çoklu format desteği (JPG, PNG, BMP, TIFF)
- Gelişmiş OCR motoru (EasyOCR + Tesseract)
- Akıllı görüntü ön işleme
- Bulanık string eşleştirme
- TCKN algoritma doğrulaması
- Detaylı raporlama sistemi

Kullanım:
    python verify_documents.py kimlik_kart.jpg basvuru_formu.jpg
"""

import argparse
import re
import sys
import os
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

import cv2
import easyocr
import numpy as np
from rapidfuzz import fuzz, process
from PIL import Image, ImageEnhance


@dataclass
class DocumentInfo:
    """Belge bilgilerini tutan veri sınıfı."""
    name: Optional[str] = None
    tckn: Optional[str] = None
    confidence: float = 0.0
    extraction_method: str = ""


@dataclass
class ValidationResult:
    """Doğrulama sonuçlarını tutan veri sınıfı."""
    is_valid: bool
    message: str
    name_similarity: float = 0.0
    tckn_match: bool = False
    details: Dict = None


class DocumentValidator:
    """
    Belge doğrulama sınıfı.
    OCR, görüntü işleme ve karşılaştırma işlemlerini yönetir.
    """
    
    # Sabit değerler
    MIN_CONFIDENCE = 0.4  # OCR güven skoru eşiği
    MIN_FUZZ_SCORE = 75   # Bulanık eşleştirme eşiği
    MAX_IMAGE_SIZE = 4000  # Maksimum görüntü boyutu
    
    # TCKN regex pattern (11 haneli sayı)
    TCKN_PATTERN = re.compile(r'^[0-9]{11}$')
    
    # Türkçe karakter mapping
    TURKISH_CHARS = {
        'ç': 'c', 'ğ': 'g', 'ı': 'i', 'ö': 'o', 'ş': 's', 'ü': 'u',
        'Ç': 'C', 'Ğ': 'G', 'İ': 'I', 'Ö': 'O', 'Ş': 'S', 'Ü': 'U'
    }
    
    def __init__(self):
        """Sınıf başlatıcısı."""
        self.reader = None
        self._initialize_ocr()
    
    def _initialize_ocr(self):
        """OCR motorunu başlatır."""
        try:
            # EasyOCR reader'ı başlat (Türkçe + İngilizce)
            self.reader = easyocr.Reader(['tr', 'en'], gpu=False)
            print("OCR motoru başarıyla başlatıldı.")
        except Exception as e:
            print(f"OCR başlatma hatası: {e}")
            self.reader = None
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Görüntüyü OCR için optimize eder.
        
        Args:
            image_path: Görüntü dosyasının yolu
            
        Returns:
            Ön işlenmiş görüntü array'i
        """
        try:
            # Görüntüyü yükle
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Görüntü yüklenemedi: {image_path}")
            
            # Boyut kontrolü ve yeniden boyutlandırma
            height, width = image.shape[:2]
            if max(height, width) > self.MAX_IMAGE_SIZE:
                scale = self.MAX_IMAGE_SIZE / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Gri tonlamaya çevir
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Gürültü azaltma (bilateral filter)
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Kontrast artırma
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            
            # Adaptif eşikleme
            thresh = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Morfolojik işlemler
            kernel = np.ones((1, 1), np.uint8)
            processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            return processed
            
        except Exception as e:
            raise ValueError(f"Görüntü işleme hatası: {str(e)}")
    
    def extract_tckn_from_text(self, text: str) -> Optional[str]:
        """
        Metinden TCKN'yi çıkarır ve doğrular.
        
        Args:
            text: OCR'dan gelen metin
            
        Returns:
            Doğrulanmış TCKN string'i veya None
        """
        # Sadece rakamları al
        digits = re.findall(r'\d+', text)
        
        for digit_group in digits:
            # 11 haneli sayı ara
            if len(digit_group) == 11 and self.TCKN_PATTERN.match(digit_group):
                # TCKN algoritma doğrulaması
                if self._validate_tckn_algorithm(digit_group):
                    return digit_group
        
        return None
    
    def _validate_tckn_algorithm(self, tckn: str) -> bool:
        """
        TCKN algoritma doğrulaması yapar.
        
        Args:
            tckn: 11 haneli TCKN
            
        Returns:
            Algoritma doğrulaması sonucu
        """
        try:
            digits = [int(d) for d in tckn]
            
            # 10. hane kontrolü
            odd_sum = sum(digits[i] for i in range(0, 9, 2))
            even_sum = sum(digits[i] for i in range(1, 8, 2))
            expected_10 = (odd_sum * 7 - even_sum) % 10
            
            if digits[9] != expected_10:
                return False
            
            # 11. hane kontrolü
            expected_11 = sum(digits[:10]) % 10
            if digits[10] != expected_11:
                return False
            
            return True
            
        except (IndexError, ValueError):
            return False
    
    def extract_name_from_text(self, text: str) -> Optional[str]:
        """
        Metinden isim bilgisini çıkarır ve temizler.
        
        Args:
            text: OCR'dan gelen metin
            
        Returns:
            Temizlenmiş isim string'i veya None
        """
        # Gereksiz kelimeleri filtrele (başlıklar ve etiketler)
        exclude_words = {
            'TÜRKİYE', 'CUMHURİYETİ', 'KİMLİK', 'KARTI', 'BAŞVURU', 'FORMU',
            'TURKEY', 'REPUBLIC', 'IDENTITY', 'CARD', 'APPLICATION', 'FORM',
            'SOYADI', 'SURNAME', 'ADI', 'GIVEN', 'NAME', 'NAMES', 'T.C.', 'TC',
            'KİMLİK', 'NO', 'IDENTITY', 'DOĞUM', 'TARİHİ', 'DATE', 'BIRTH',
            'CİNSİYETİ', 'GENDER', 'BELGE', 'DOCUMENT', 'UYRUK', 'NATIONALITY',
            'GEÇERLİLİK', 'VALID', 'UNTIL', 'İMZA', 'SIGNATURE', 'ÖRNEKTİR'
        }
        
        # Türkçe karakterleri koru, sadece harf ve boşluk bırak
        cleaned = re.sub(r'[^a-zA-ZçğıöşüÇĞIİÖŞÜ\s]', '', text)
        
        # Fazla boşlukları temizle
        cleaned = ' '.join(cleaned.split())
        
        # Gereksiz kelimeleri çıkar
        words = cleaned.split()
        filtered_words = [word for word in words if word.upper() not in exclude_words]
        
        # En az 2 kelime olmalı (Ad Soyad)
        if len(filtered_words) >= 2 and len(' '.join(filtered_words)) >= 4:
            # İlk iki kelimeyi al (Ad Soyad)
            name = ' '.join(filtered_words[:2])
            return name.strip()
        
        return None
    
    def normalize_name(self, name: str) -> str:
        """
        İsmi karşılaştırma için normalize eder.
        
        Args:
            name: Orijinal isim
            
        Returns:
            Normalize edilmiş isim
        """
        # Küçük harfe çevir
        normalized = name.lower()
        
        # Türkçe karakterleri İngilizce karşılıklarına çevir
        for turkish, english in self.TURKISH_CHARS.items():
            normalized = normalized.replace(turkish, english)
        
        # Fazla boşlukları temizle
        normalized = ' '.join(normalized.split())
        
        return normalized

    # Yardımcı: EasyOCR bbox -> dikdörtgen
    def _rect_from_bbox(self, bbox: List[List[float]]) -> Tuple[int, int, int, int]:
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x_min, x_max = int(min(xs)), int(max(xs))
        y_min, y_max = int(min(ys)), int(max(ys))
        return x_min, y_min, x_max, y_max

    def _center_of_rect(self, rect: Tuple[int, int, int, int]) -> Tuple[int, int]:
        x_min, y_min, x_max, y_max = rect
        return (x_min + x_max) // 2, (y_min + y_max) // 2

    def _is_text_value(self, text: str) -> bool:
        text_stripped = text.strip()
        return len(text_stripped) >= 2 and all(ch.isalpha() or ch.isspace() for ch in text_stripped)

    def _find_value_near_label(self, ocr_results: List, label_keywords: List[str]) -> Optional[str]:
        """
        Bir etiket (ör. SOYADI / SURNAME) için, etiket kutusuna en yakın metin değerini döndürür.
        """
        label_candidates: List[Tuple[Tuple[int,int,int,int], float]] = []
        for (bbox, text, confidence) in ocr_results:
            if confidence < self.MIN_CONFIDENCE:
                continue
            upper = text.upper()
            if any(k in upper for k in label_keywords):
                label_candidates.append((self._rect_from_bbox(bbox), confidence))

        if not label_candidates:
            return None

        # En güvenilir etiketi kullan
        label_rect, _ = max(label_candidates, key=lambda rc: rc[1])
        label_cx, label_cy = self._center_of_rect(label_rect)

        best_text = None
        best_dist = 10**9
        for (bbox, text, confidence) in ocr_results:
            if confidence < self.MIN_CONFIDENCE:
                continue
            if not self._is_text_value(text):
                continue
            # Etiket metninin kendisini veya başlıkları değer olarak kullanmayalım
            upper_text = text.upper()
            if any(k in upper_text for k in label_keywords):
                continue
            if any(h in upper_text for h in [
                'TÜRKİYE', 'CUMHURİYETİ', 'IDENTITY', 'CARD', 'T.C', 'TC',
                'SOYADI', 'SURNAME', 'ADI', 'GIVEN', 'NAME', 'NAMES']):
                continue
            value_rect = self._rect_from_bbox(bbox)
            vx, vy = self._center_of_rect(value_rect)
            # Tercihen etiketin sağında veya hemen altında olan değerleri seç
            to_right = vx >= label_cx
            below = vy >= label_cy
            if not (to_right or below):
                continue
            dist = abs(vx - label_cx) + abs(vy - label_cy)
            if dist < best_dist:
                best_dist = dist
                best_text = text.strip()

        return best_text
    
    def extract_document_info(self, image_path: str) -> DocumentInfo:
        """
        Görüntüden belge bilgilerini çıkarır.
        
        Args:
            image_path: Görüntü dosyasının yolu
            
        Returns:
            Çıkarılan belge bilgileri
        """
        if not self.reader:
            raise ValueError("OCR motoru başlatılamadı")
        
        try:
            # Görüntüyü ön işle
            processed_image = self.preprocess_image(image_path)
            
            # OCR işlemi
            results = self.reader.readtext(processed_image)
            
            extracted_name = None
            extracted_tckn = None
            best_confidence = 0.0
            
            # Kimlik kartı için özel işlem
            if "kimlik" in image_path.lower() or "id" in image_path.lower():
                extracted_name = self._extract_id_card_name(results)
            else:
                # Form için özel işlem
                extracted_name = self._extract_form_name(results)
                
                # TCKN ara
                for (bbox, text, confidence) in results:
                    if confidence < self.MIN_CONFIDENCE:
                        continue
                    tckn = self.extract_tckn_from_text(text)
                    if tckn:
                        extracted_tckn = tckn
                        best_confidence = max(best_confidence, confidence)
                        break
            
            # TCKN bulunamadıysa tüm sonuçlarda ara
            if not extracted_tckn:
                for (bbox, text, confidence) in results:
                    if confidence < self.MIN_CONFIDENCE:
                        continue
                    tckn = self.extract_tckn_from_text(text)
                    if tckn:
                        extracted_tckn = tckn
                        best_confidence = max(best_confidence, confidence)
                        break
            
            return DocumentInfo(
                name=extracted_name,
                tckn=extracted_tckn,
                confidence=best_confidence,
                extraction_method="OCR"
            )
            
        except Exception as e:
            print(f"OCR işlemi hatası: {str(e)}", file=sys.stderr)
            return DocumentInfo()
    
    def _extract_id_card_name(self, ocr_results: List) -> Optional[str]:
        """
        Kimlik kartından ad ve soyadı, etiketlerine göre çıkarır ve 'ADI SOYADI' sırası ile döndürür.
        """
        # Öncelik: Etiket bazlı yakalama
        given = self._find_value_near_label(ocr_results, ["ADI", "GIVEN NAME", "GIVEN NAME(S)", "GIVEN NAMES"])
        surname = self._find_value_near_label(ocr_results, ["SOYADI", "SURNAME"])

        def clean_token(t: Optional[str]) -> Optional[str]:
            if not t:
                return None
            # Sadece harf ve boşluk
            filtered = re.sub(r"[^A-Za-zÇĞİÖŞÜçğıöşü\s]", "", t).strip()
            return filtered if filtered else None

        given = clean_token(given)
        surname = clean_token(surname)

        if given and surname:
            return f"{given.upper()} {surname.upper()}"

        # Geriye dönüş: mevcut basit mantık
        given_name = None
        surname_name = None
        for (bbox, text, confidence) in ocr_results:
            if confidence < self.MIN_CONFIDENCE:
                continue
            val = clean_token(text)
            if not val or not val.replace(" ", "").isalpha():
                continue
            up = val.upper()
            if not surname_name and up not in ["SOYADI", "SURNAME", "ADI", "GIVEN", "NAMES", "TÜRKİYE", "CUMHURİYETİ"] and len(up) >= 3:
                surname_name = up
                continue
            if not given_name and up not in ["SOYADI", "SURNAME", "ADI", "GIVEN", "NAMES", "TÜRKİYE", "CUMHURİYETİ"] and len(up) >= 2:
                given_name = up
                continue
        if given_name and surname_name:
            return f"{given_name} {surname_name}"
        return given_name or surname_name
    
    def _extract_form_name(self, ocr_results: List) -> Optional[str]:
        """
        Başvuru formundan ad soyad bilgisini çıkarır.
        
        Args:
            ocr_results: OCR sonuçları
            
        Returns:
            Çıkarılan ad soyad
        """

        
        # Öncelik: 'Ad Soyad' etiketine göre değeri al
        value = self._find_value_near_label(ocr_results, ["AD SOYAD", "AD SOYAD:"])
        if value:
            # Temizle ve sadece iki kelimeyi bırak
            clean = re.sub(r"[^A-Za-zÇĞİÖŞÜçğıöşü\s]", "", value).strip()
            parts = [p for p in clean.split() if p.isalpha()]
            if len(parts) >= 2:
                return f"{parts[0]} {parts[1]}"
            elif len(parts) == 1:
                return parts[0]

        # Geriye dönüş: en iyi 2+ kelimeli satırı seç
        header_words = {'BAŞVURU FORMU', 'AD SOYAD', 'TC KİMLİK', 'KİMLİK NUMARASI','TELEFON','E-POSTA','ADRES','İMZA','TARİH'}
        best_candidate = None
        best_confidence = 0.0
        for (bbox, text, confidence) in ocr_results:
            if confidence < self.MIN_CONFIDENCE:
                continue
            text_clean = text.strip()
            if (text_clean.upper() not in header_words and ' ' in text_clean and len(text_clean.split()) >= 2):
                words = [w for w in text_clean.split() if w.isalpha()]
                if len(words) >= 2 and confidence > best_confidence:
                    best_candidate = f"{words[0]} {words[1]}"
                    best_confidence = confidence
        return best_candidate
    
    def compare_documents(self, id_data: DocumentInfo, form_data: DocumentInfo) -> ValidationResult:
        """
        İki belgeyi katı kurallarla karşılaştırır: TCKN tam eşleşmeli,
        soyad eşleşmesi zorunlu, ad için güçlü benzerlik aranır.
        """
        details = {
            "id_name": id_data.name,
            "form_name": form_data.name,
            "id_tckn": id_data.tckn,
            "form_tckn": form_data.tckn,
            "name_similarity": 0.0,
            "tckn_match": False
        }

        # TCKN karşılaştırması
        tckn_match = bool(id_data.tckn and form_data.tckn and id_data.tckn == form_data.tckn)
        details["tckn_match"] = tckn_match

        name_similarity = 0.0
        name_match = False
        if id_data.name and form_data.name:
            id_norm = self.normalize_name(id_data.name)
            form_norm = self.normalize_name(form_data.name)

            id_tokens = [t for t in id_norm.split() if len(t) > 1]
            form_tokens = [t for t in form_norm.split() if len(t) > 1]
            if id_tokens and form_tokens:
                id_given, id_surname = id_tokens[0], id_tokens[-1]
                form_given, form_surname = form_tokens[0], form_tokens[-1]

                # Soyad tam eşleşme ya da çok yüksek benzerlik
                surname_ratio = fuzz.ratio(id_surname, form_surname)
                surname_ok = (id_surname == form_surname) or surname_ratio >= 90

                # Ad için yüksek benzerlik veya içerme
                given_ratio = max(
                    fuzz.ratio(id_given, form_given),
                    fuzz.partial_ratio(id_given, form_given)
                )
                given_ok = given_ratio >= 85

                # Toplam isim benzerliği (ağırlıklı)
                full_ratio = max(
                    fuzz.ratio(" ".join(id_tokens), " ".join(form_tokens)),
                    fuzz.token_sort_ratio(" ".join(id_tokens), " ".join(form_tokens))
                )
                name_similarity = max(full_ratio, int(0.6 * surname_ratio + 0.4 * given_ratio))
                name_match = surname_ok and (name_similarity >= max(self.MIN_FUZZ_SCORE, 85))

        details["name_similarity"] = name_similarity

        if tckn_match and name_match:
            message = "Olumlu - Tüm bilgiler eşleşiyor"
            is_valid = True
        elif not tckn_match:
            message = "Belgedeki TC Kimlik Numarası Hatalı"
            is_valid = False
        else:
            message = f"Belgedeki Ad Soyad Hatalı (Benzerlik: %{name_similarity:.1f})"
            is_valid = False

        return ValidationResult(
            is_valid=is_valid,
            message=message,
            name_similarity=name_similarity,
            tckn_match=tckn_match,
            details=details
        )
    
    def validate_file_format(self, file_path: str) -> bool:
        """
        Dosya formatını kontrol eder.
        
        Args:
            file_path: Dosya yolu
            
        Returns:
            Format geçerliliği
        """
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        _, ext = os.path.splitext(file_path.lower())
        return ext in valid_extensions


def main():
    """Ana uygulama fonksiyonu."""
    parser = argparse.ArgumentParser(
        description="Türk Kimlik Kartı ve Başvuru Formu Doğrulama Sistemi v2.0"
    )
    parser.add_argument(
        "id_image", 
        help="Kimlik kartı görüntü dosyasının yolu"
    )
    parser.add_argument(
        "form_image", 
        help="Başvuru formu görüntü dosyasının yolu"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Detaylı çıktı göster"
    )
    
    try:
        args = parser.parse_args()
        
        # Dosya varlığını kontrol et
        if not os.path.exists(args.id_image):
            raise FileNotFoundError(f"Kimlik kartı dosyası bulunamadı: {args.id_image}")
        
        if not os.path.exists(args.form_image):
            raise FileNotFoundError(f"Başvuru formu dosyası bulunamadı: {args.form_image}")
        
        # Validator'ı başlat
        validator = DocumentValidator()
        
        # Dosya formatlarını kontrol et
        if not validator.validate_file_format(args.id_image):
            raise ValueError(f"Geçersiz kimlik kartı formatı: {args.id_image}")
        
        if not validator.validate_file_format(args.form_image):
            raise ValueError(f"Geçersiz form formatı: {args.form_image}")
        
        # Bilgileri çıkar
        print("Kimlik kartı işleniyor...", file=sys.stderr)
        id_data = validator.extract_document_info(args.id_image)
        
        print("Başvuru formu işleniyor...", file=sys.stderr)
        form_data = validator.extract_document_info(args.form_image)
        
        # Hata kontrolü
        if not id_data.name or not id_data.tckn:
            print("Kimlik kartından gerekli bilgiler çıkarılamadı.", file=sys.stderr)
            sys.exit(1)
            
        if not form_data.name or not form_data.tckn:
            print("Başvuru formundan gerekli bilgiler çıkarılamadı.", file=sys.stderr)
            sys.exit(1)
        
        # Karşılaştır
        result = validator.compare_documents(id_data, form_data)
        
        # Sonucu yazdır
        print(result.message)
        
        # Detaylı çıktı
        if args.verbose:
            print(f"\nSonuç:")
            print(f"Kimlik Kartı: {id_data.name} - {id_data.tckn}")
            print(f"Başvuru Formu: {form_data.name} - {form_data.tckn}")
            print(f"İsim Benzerliği: %{result.name_similarity:.1f}")
            print(f"TCKN: {'Eşleşti' if result.tckn_match else 'Eşleşmedi'}")
            print(f"OCR Güven Skoru: %{max(id_data.confidence, form_data.confidence):.1f}")
        
    except (ValueError, FileNotFoundError) as e:
        print(f"Hata: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹İşlem kullanıcı tarafından durduruldu.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Beklenmeyen hata: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

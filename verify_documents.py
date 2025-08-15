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
            print("✅ OCR motoru başarıyla başlatıldı.")
        except Exception as e:
            print(f"⚠️ OCR başlatma hatası: {e}")
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
        Kimlik kartından ad ve soyadı ayrı ayrı çıkarır.
        
        Args:
            ocr_results: OCR sonuçları
            
        Returns:
            Birleştirilmiş ad soyad
        """
        given_name = None
        surname = None
        
        # Tüm OCR sonuçlarını analiz et
        for (bbox, text, confidence) in ocr_results:
            if confidence < self.MIN_CONFIDENCE:
                continue
            
            text_upper = text.upper().strip()
            
            # Soyadı ara (KARACA gibi)
            if not surname and len(text_upper) >= 3:
                # Sadece harflerden oluşan, başlık olmayan metinler
                if (text_upper.isalpha() and 
                    text_upper not in ['SOYADI', 'SURNAME', 'ADI', 'GIVEN', 'NAMES', 'TÜRKİYE', 'CUMHURİYETİ']):
                    surname = text_upper
                    continue
            
            # Ad ara (ALİ gibi)
            if not given_name and len(text_upper) >= 2:
                if (text_upper.isalpha() and 
                    text_upper not in ['SOYADI', 'SURNAME', 'ADI', 'GIVEN', 'NAMES', 'TÜRKİYE', 'CUMHURİYETİ']):
                    given_name = text_upper
                    continue
        
        # Ad ve soyadı birleştir
        if given_name and surname:
            # Soyadı önce, ad sonra (Türk geleneği)
            return f"{surname} {given_name}"
        elif surname:
            return surname
        elif given_name:
            return given_name
        
        return None
    
    def _extract_form_name(self, ocr_results: List) -> Optional[str]:
        """
        Başvuru formundan ad soyad bilgisini çıkarır.
        
        Args:
            ocr_results: OCR sonuçları
            
        Returns:
            Çıkarılan ad soyad
        """

        
        # Başlık kelimeleri (tam eşleşme için)
        header_words = {
            'BAŞVURU FORMU', 'AD SOYAD', 'TC KİMLİK', 'KİMLİK NUMARASI',
            'TELEFON', 'E-POSTA', 'ADRES', 'İMZA', 'TARİH'
        }
        
        # En iyi adayı bul
        best_candidate = None
        best_confidence = 0.0
        
        for (bbox, text, confidence) in ocr_results:
            if confidence < self.MIN_CONFIDENCE:
                continue
            
            text_clean = text.strip()
            
            # Başlık kelimeleri değilse ve 2+ kelime içeriyorsa
            if (text_clean.upper() not in header_words and 
                ' ' in text_clean and 
                len(text_clean.split()) >= 2):
                
                words = text_clean.split()
                # İlk iki kelime sadece harflerden oluşmalı
                if all(word.isalpha() for word in words[:2]):
                    # Bu aday daha iyi mi?
                    if confidence > best_confidence:
                        best_candidate = ' '.join(words[:2])
                        best_confidence = confidence
                        print(f"🔍 Yeni aday bulundu: {best_candidate} (güven: %{confidence*100:.1f})", file=sys.stderr)
        
        if best_candidate:
            return best_candidate
        
        return None
    
    def compare_documents(self, id_data: DocumentInfo, form_data: DocumentInfo) -> ValidationResult:
        """
        İki belgeyi karşılaştırır.
        
        Args:
            id_data: Kimlik kartı bilgileri
            form_data: Form bilgileri
            
        Returns:
            Karşılaştırma sonucu
        """
        details = {
            "id_name": id_data.name,
            "form_name": form_data.name,
            "id_tckn": id_data.tckn,
            "form_tckn": form_data.tckn,
            "name_similarity": 0.0,
            "tckn_match": False
        }
        
        # TCKN karşılaştırması (tam eşleşme)
        tckn_match = False
        if id_data.tckn and form_data.tckn:
            tckn_match = id_data.tckn == form_data.tckn
            details["tckn_match"] = tckn_match
        
        # İsim karşılaştırması (bulanık eşleşme)
        name_similarity = 0.0
        name_match = False
        
        if id_data.name and form_data.name:
            # Normalize edilmiş isimleri karşılaştır
            norm_id_name = self.normalize_name(id_data.name)
            norm_form_name = self.normalize_name(form_data.name)
            
            # Farklı karşılaştırma yöntemleri
            ratio_score = fuzz.ratio(norm_id_name, norm_form_name)
            partial_score = fuzz.partial_ratio(norm_id_name, norm_form_name)
            token_sort_score = fuzz.token_sort_ratio(norm_id_name, norm_form_name)
            
            # En yüksek skoru al
            name_similarity = max(ratio_score, partial_score, token_sort_score)
            name_match = name_similarity >= self.MIN_FUZZ_SCORE
            
            details["name_similarity"] = name_similarity
        
        # Sonuç belirleme
        if tckn_match and name_match:
            message = "Olumlu - Tüm bilgiler eşleşiyor"
            is_valid = True
        elif not name_match:
            message = f"Belgedeki Ad Soyad Hatalı (Benzerlik: %{name_similarity:.1f})"
            is_valid = False
        elif not tckn_match:
            message = "Belgedeki TC Kimlik Numarası Hatalı"
            is_valid = False
        else:
            message = "Belge bilgileri çıkarılamadı"
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
            print(f"\n Sonuç:")
            print(f"   Kimlik Kartı: {id_data.name} - {id_data.tckn}")
            print(f"   Başvuru Formu: {form_data.name} - {form_data.tckn}")
            print(f"   İsim Benzerliği: %{result.name_similarity:.1f}")
            print(f"   TCKN: {'Eşleşti' if result.tckn_match else 'Eşleşmedi'}")
            print(f"   OCR Güven Skoru: %{max(id_data.confidence, form_data.confidence):.1f}")
        
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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Örnek Başvuru Formu Oluşturma Modülü
PDF ve JPG formatlarında örnek başvuru formları üretir.

Author: Mustafa Kursad Baser
Version: 1.0.0
"""

import argparse
import os
import random
import sys
from datetime import datetime, timedelta
from typing import Dict

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors


# Türkçe isim veritabanı (Ad Soyad)
TURKISH_NAMES = [
    "Ali Karaca", "Ahmet Yılmaz", "Fatma Özkan", "Mehmet Çelik", "Ayşe Şahin", "Mustafa Öztürk"
]

# Mobil operatör kodları
PHONE_PREFIXES = ["532", "533", "534", "535", "536", "537", "538", "539", "540", "541"]

# E-posta domaini
EMAIL_DOMAIN = "mustafabaser.net"

# Türkiye adres veritabanı
ADDRESSES = [
    "Atatürk Mahallesi, Cumhuriyet Caddesi No:123, Kadıköy/İstanbul",
    "Çankaya Mahallesi, İstiklal Sokak No:45, Çankaya/Ankara",
    "Alsancak Mahallesi, Kıbrıs Şehitleri Caddesi No:67, Konak/İzmir",
    "Kızılay Mahallesi, Atatürk Bulvarı No:89, Çankaya/Ankara",
    "Beşiktaş Mahallesi, Barbaros Bulvarı No:12, Beşiktaş/İstanbul",
    "Karşıyaka Mahallesi, Atatürk Caddesi No:34, Karşıyaka/İzmir",
    "Nilüfer Mahallesi, FSM Bulvarı No:56, Nilüfer/Bursa",
    "Tepebaşı Mahallesi, İnönü Caddesi No:78, Tepebaşı/Eskişehir"
]


def generate_tckn() -> str:
    """
    Geçerli TC Kimlik Numarası algoritmasına uygun rastgele TCKN üretir.
    
    Returns:
        str: 11 haneli geçerli TC Kimlik Numarası
    """
    # İlk 9 haneyi rastgele oluştur (1 ile başlamalı)
    digits = [random.randint(1, 9)]
    digits.extend([random.randint(0, 9) for _ in range(8)])
    
    # 10. hane: (1+3+5+7+9)*7 + (2+4+6+8)*9 mod 10
    odd_sum = sum(digits[i] for i in range(0, 9, 2))
    even_sum = sum(digits[i] for i in range(1, 8, 2))
    digit_10 = (odd_sum * 7 - even_sum) % 10
    
    # 11. hane: İlk 10 hanenin toplamı mod 10
    digit_11 = sum(digits + [digit_10]) % 10
    
    return ''.join(map(str, digits + [digit_10, digit_11]))


def generate_mock_data(tckn: str = None) -> Dict[str, str]:
    """
    Gerçekçi başvuru verisi oluşturur.
    
    Args:
        tckn: TC Kimlik Numarası (None ise rastgele üretilir)
        
    Returns:
        Dict[str, str]: Başvuru formu için gerekli tüm alanlar
    """
    # İsim seçimi
    full_name = random.choice(TURKISH_NAMES)
    
    # TCKN (verilmişse kullan, yoksa üret)
    if tckn is None:
        tckn = generate_tckn()
    
    # Telefon numarası oluşturma
    phone_prefix = random.choice(PHONE_PREFIXES)
    phone_suffix = ''.join([str(random.randint(0, 9)) for _ in range(7)])
    phone = f"0{phone_prefix} {phone_suffix[:3]} {phone_suffix[3:5]} {phone_suffix[5:]}"
    
    # E-posta adresi oluşturma (şirket domaini kullanılır)
    name_parts = full_name.lower().replace(' ', '').replace('ç', 'c').replace('ğ', 'g').replace('ı', 'i').replace('ö', 'o').replace('ş', 's').replace('ü', 'u')
    email = f"{name_parts}@{EMAIL_DOMAIN}"
    
    # Adres seçimi
    address = random.choice(ADDRESSES)
    
    # Tarih oluşturma (son 30 gün içinde)
    random_days = random.randint(0, 30)
    date = datetime.now() - timedelta(days=random_days)
    date_str = date.strftime("%d.%m.%Y")
    
    return {
        "full_name": full_name,
        "tckn": tckn,
        "phone": phone,
        "email": email,
        "address": address,
        "date": date_str
    }


def create_pdf_form(data: Dict[str, str], output_path: str):
    """
    PDF formatında örnek başvuru formu oluşturur.
    
    Args:
        data: Başvuru verileri
        output_path: PDF dosya yolu
    """
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    story = []
    
    # Türkçe karakter desteği için font yükleme
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    
    # Windows Arial fontu yükleme
    try:
        pdfmetrics.registerFont(TTFont('Arial', 'C:/Windows/Fonts/arial.ttf'))
        pdfmetrics.registerFont(TTFont('Arial-Bold', 'C:/Windows/Fonts/arialbd.ttf'))
        font_name = 'Arial'
        font_bold = 'Arial-Bold'
    except Exception as e:
        try:
            # Linux DejaVu fontu yükleme
            pdfmetrics.registerFont(TTFont('DejaVu', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
            pdfmetrics.registerFont(TTFont('DejaVu-Bold', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
            font_name = 'DejaVu'
            font_bold = 'DejaVu-Bold'
        except Exception as e2:
            # Varsayılan font kullanımı
            font_name = 'Helvetica'
            font_bold = 'Helvetica-Bold'
    
    # Stil tanımlamaları
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
        alignment=1,  # Ortalı
        fontName=font_bold
    )
    
    field_style = ParagraphStyle(
        'FieldStyle',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=6,
        fontName=font_name
    )
    
    label_style = ParagraphStyle(
        'LabelStyle',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=2,
        fontName=font_bold,
        textColor=colors.darkblue
    )
    
    # Başlık
    story.append(Paragraph("BAŞVURU FORMU", title_style))
    story.append(Spacer(1, 20))
    
    # Form alanları
    fields = [
        ("Ad Soyad:", data["full_name"]),
        ("TC Kimlik Numarası:", data["tckn"]),
        ("Telefon:", data["phone"]),
        ("E-Posta:", data["email"]),
        ("Adres:", data["address"])
    ]
    
    for label, value in fields:
        story.append(Paragraph(label, label_style))
        story.append(Paragraph(value, field_style))
        story.append(Spacer(1, 10))
    
    # İmza ve tarih bölümü
    story.append(Spacer(1, 30))
    
    # İmza alanı
    signature_table = Table([
        ["İmza:", "_________________________"],
        ["Tarih:", data["date"]]
    ], colWidths=[3*cm, 8*cm])
    
    signature_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), font_bold),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
    ]))
    
    story.append(signature_table)
    
    # PDF oluşturma
    doc.build(story)


def create_jpg_form(data: Dict[str, str], output_path: str):
    """
    JPG formatında örnek başvuru formu oluşturur.
    
    Args:
        data: Başvuru verileri
        output_path: JPG dosya yolu
    """
    # A4 boyutunda beyaz arka plan (300 DPI)
    width, height = int(21.0 * 300 / 2.54), int(29.7 * 300 / 2.54)  # A4 boyutu
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Türkçe karakter desteği için font yükleme
    try:
        # Windows Arial fontu
        title_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 72)
        label_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 48)
        text_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 40)
    except:
        try:
            # Linux DejaVu fontu
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 72)
            label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 48)
            text_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40)
        except:
            # Varsayılan font
            title_font = ImageFont.load_default()
            label_font = ImageFont.load_default()
            text_font = ImageFont.load_default()
    
    # Başlık
    title = "BAŞVURU FORMU"
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (width - title_width) // 2
    draw.text((title_x, 200), title, fill='black', font=title_font)
    
    # Form alanları
    y_position = 400
    line_height = 120
    
    fields = [
        ("Ad Soyad:", data["full_name"]),
        ("TC Kimlik Numarası:", data["tckn"]),
        ("Telefon:", data["phone"]),
        ("E-Posta:", data["email"]),
        ("Adres:", data["address"])
    ]
    
    for label, value in fields:
        # Etiket
        draw.text((200, y_position), label, fill='darkblue', font=label_font)
        
        # Değer
        draw.text((200, y_position + 50), value, fill='black', font=text_font)
        # Altına çizgi
        draw.line([(200, y_position + 100), (width - 200, y_position + 100)], fill='black', width=4)
        
        y_position += line_height
    
    # İmza ve tarih bölümü
    y_position += 100
    
    # İmza alanı
    draw.text((200, y_position), "İmza:", fill='darkblue', font=label_font)
    draw.line([(400, y_position + 50), (800, y_position + 50)], fill='black', width=4)
    
    # Tarih
    draw.text((200, y_position + 120), "Tarih:", fill='darkblue', font=label_font)
    draw.text((400, y_position + 120), data["date"], fill='black', font=text_font)
    
    # Çerçeve
    draw.rectangle([(100, 100), (width - 100, height - 100)], outline='black', width=6)
    
    # Kaydetme
    image.save(output_path, 'JPEG', quality=95)


def main():
    """
    Ana uygulama fonksiyonu.
    Komut satırı argümanlarını işler ve başvuru formları oluşturur.
    """
    parser = argparse.ArgumentParser(
        description="Başvuru formu örneği oluşturma uygulaması"
    )
    parser.add_argument(
        "--format", 
        choices=["pdf", "jpg", "both"], 
        default="both",
        help="Çıktı formatı seçimi (varsayılan: both)"
    )
    parser.add_argument(
        "--output", 
        default="basvuru_formu",
        help="Çıktı dosya adı (uzantısız)"
    )
    parser.add_argument(
        "--count", 
        type=int, 
        default=1,
        help="Oluşturulacak form sayısı (varsayılan: 1)"
    )
    parser.add_argument(
        "--tckn", 
        type=str,
        default=None,
        help="TC Kimlik Numarası (belirtilmezse rastgele üretilir)"
    )
    
    args = parser.parse_args()
    
    # Çıktı klasörü oluşturma
    output_folder = "application_forms"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"'{output_folder}' klasörü oluşturuldu.")
    else:
        print(f"'{output_folder}' klasörü bulundu.")
    
    try:
        for i in range(args.count):
            # Başvuru verisi oluşturma
            data = generate_mock_data(args.tckn)
            
            # Dosya adları
            if args.count > 1:
                base_name = f"{args.output}_{i+1}"
            else:
                base_name = args.output
            
            print(f"Form {i+1} oluşturuluyor...")
            print(f"Ad Soyad: {data['full_name']}")
            print(f"TCKN: {data['tckn']}")
            print(f"Telefon: {data['phone']}")
            print(f"E-posta: {data['email']}")
            print(f"Tarih: {data['date']}")
            print("-" * 50)
            
            # Formatlara göre oluşturma
            if args.format in ["pdf", "both"]:
                pdf_path = os.path.join(output_folder, f"{base_name}.pdf")
                create_pdf_form(data, pdf_path)
                print(f"PDF oluşturuldu: {pdf_path}")
            
            if args.format in ["jpg", "both"]:
                jpg_path = os.path.join(output_folder, f"{base_name}.jpg")
                create_jpg_form(data, jpg_path)
                print(f"JPG oluşturuldu: {jpg_path}")
        
        print(f"\n{args.count} adet başvuru formu başarıyla oluşturuldu!")
        print(f"Dosyalar '{output_folder}' klasörüne kaydedildi.")
        
    except Exception as e:
        print(f"Hata: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

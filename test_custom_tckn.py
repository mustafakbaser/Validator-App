#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom TCKN üzerinden örnek başvuru formu oluşturma modülü

Author: Mustafa Kursad Baser
Version: 1.0.0
"""

import os
import sys

def run_form_generator(tckn: str, output_name: str = "basvuru_formu"):
    """
    generate_application_form.py dosyasını belirtilen TCKN ile çalıştırır.
    
    Args:
        tckn: TC Kimlik Numarası
        output_name: Çıktı dosya adı
    """
    try:
        # Komut oluştur
        cmd = f'python generate_application_form.py --tckn {tckn} --output {output_name} --format both'
        
        print(f"Komut çalıştırılıyor: {cmd}")
        print("-" * 60)
        
        # Komutu çalıştır
        result = os.system(cmd)
        
        if result == 0:
            print(f"\n({tckn}) Form TCKN ile oluşturuldu!")
        else:
            print(f"\nForm oluşturulurken hata oluştu!")
            
    except Exception as e:
        print(f"Hata: {str(e)}")

if __name__ == "__main__":
    # Rastgele TCKN yerine custom TCKN ile form oluştur:
    custom_tckn = "01234567890"
    run_form_generator(custom_tckn, "basvuru_formu")
    
    print("Dosyalar oluşturuldu.")
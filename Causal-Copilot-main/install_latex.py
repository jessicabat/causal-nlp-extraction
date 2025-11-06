#!/usr/bin/env python3
"""
LaTeX Installation Helper for Causality-Copilot

This script helps install LaTeX distribution and related packages for document compilation.
"""

import platform
import subprocess
import sys
import os

def run_command(command, description):
    """Run a system command and return success status"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ {description} failed:")
            if result.stderr.strip():
                print(f"   Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ {description} error: {e}")
        return False

def check_latex():
    """Check if LaTeX is already installed"""
    try:
        result = subprocess.run(['pdflatex', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ LaTeX is already installed and accessible")
            version_line = result.stdout.split('\n')[0]
            print(f"   Version: {version_line}")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ LaTeX not found in PATH")
    return False

def install_latex_macos():
    """Install LaTeX on macOS"""
    print("🍎 Detected macOS")
    
    # Check for Homebrew
    try:
        subprocess.run(['brew', '--version'], capture_output=True, check=True)
        print("✅ Homebrew detected")
        
        # Offer options
        print("\n📋 LaTeX Installation Options for macOS:")
        print("1. BasicTeX (lightweight, ~100MB) - Recommended")
        print("2. MacTeX (full distribution, ~4GB)")
        
        choice = input("Choose option (1 or 2): ").strip()
        
        if choice == "1":
            return run_command('brew install --cask basictex', 'Installing BasicTeX via Homebrew')
        elif choice == "2":
            return run_command('brew install --cask mactex', 'Installing MacTeX via Homebrew')
        else:
            print("❌ Invalid choice")
            return False
            
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("❌ Homebrew not found")
        print("   Please install Homebrew first:")
        print("   /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
        return False

def install_latex_linux():
    """Install LaTeX on Linux"""
    print("🐧 Detected Linux")
    
    # Detect Linux distribution
    try:
        with open('/etc/os-release', 'r') as f:
            os_info = f.read().lower()
    except FileNotFoundError:
        print("❌ Could not detect Linux distribution")
        return False
    
    if 'ubuntu' in os_info or 'debian' in os_info:
        print("🔵 Detected Ubuntu/Debian")
        return (run_command('sudo apt-get update', 'Updating package list') and
                run_command('sudo apt-get install -y texlive-latex-base texlive-latex-extra texlive-fonts-recommended', 
                           'Installing TeX Live'))
    
    elif 'centos' in os_info or 'rhel' in os_info or 'amazon' in os_info:
        print("🔴 Detected CentOS/RHEL/Amazon Linux")
        return run_command('sudo yum install -y texlive-latex texlive-latex-extra', 'Installing TeX Live')
    
    elif 'fedora' in os_info:
        print("💙 Detected Fedora")
        return run_command('sudo dnf install -y texlive-latex texlive-latex-extra', 'Installing TeX Live')
    
    else:
        print("❌ Unsupported Linux distribution")
        print("   Please install TeX Live manually using your distribution's package manager")
        return False

def install_latex_windows():
    """Install LaTeX on Windows"""
    print("🪟 Detected Windows")
    
    # Check for chocolatey
    try:
        subprocess.run(['choco', '--version'], capture_output=True, check=True)
        print("✅ Chocolatey detected")
        
        print("\n📋 LaTeX Installation Options for Windows:")
        print("1. MiKTeX (recommended for Windows)")
        print("2. TeX Live")
        
        choice = input("Choose option (1 or 2): ").strip()
        
        if choice == "1":
            return run_command('choco install miktex', 'Installing MiKTeX via Chocolatey')
        elif choice == "2":
            return run_command('choco install texlive', 'Installing TeX Live via Chocolatey')
        else:
            print("❌ Invalid choice")
            return False
            
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("❌ Chocolatey not found")
        print("   Please download and install MiKTeX manually from: https://miktex.org/download")
        print("   Or install Chocolatey first: https://chocolatey.org/install")
        return False

def install_tinytex():
    """Install TinyTeX (lightweight LaTeX distribution)"""
    print("📦 Installing TinyTeX (lightweight LaTeX distribution)...")
    
    system = platform.system()
    
    if system == "Darwin":  # macOS
        command = 'curl -sL "https://yihui.org/tinytex/install-bin-unix.sh" | sh'
    elif system == "Linux":
        command = 'curl -sL "https://yihui.org/tinytex/install-bin-unix.sh" | sh'
    elif system == "Windows":
        command = 'powershell -Command "iwr -useb https://yihui.org/tinytex/install-bin-windows.ps1 | iex"'
    else:
        print(f"❌ Unsupported platform: {system}")
        return False
    
    if run_command(command, 'Installing TinyTeX'):
        # Add to PATH
        home = os.path.expanduser("~")
        if system in ["Darwin", "Linux"]:
            tinytex_path = f"{home}/.TinyTeX/bin/x86_64-linux" if system == "Linux" else f"{home}/.TinyTeX/bin/universal-darwin"
            print(f"💡 Add to your PATH: export PATH=\"$PATH:{tinytex_path}\"")
        return True
    return False

def install_python_latex_packages():
    """Install Python packages for LaTeX integration"""
    print("🐍 Installing Python packages for LaTeX integration...")
    
    latex_packages = [
        'pylatex>=1.4.0',
        'nbconvert>=7.0.0',
        'pandoc>=2.3.0',
        'jupyter>=1.0.0',
    ]
    
    success_count = 0
    for package in latex_packages:
        if run_command(f'pip install "{package}"', f'Installing {package}'):
            success_count += 1
    
    print(f"📊 Python LaTeX packages: {success_count}/{len(latex_packages)} installed successfully")
    return success_count > 0

def install_additional_latex_packages():
    """Install additional LaTeX packages commonly needed"""
    print("📚 Installing additional LaTeX packages...")
    
    # Check if tlmgr is available
    try:
        subprocess.run(['tlmgr', '--version'], capture_output=True, check=True)
        packages = [
            'fancyhdr',
            'geometry', 
            'amsmath',
            'amsfonts',
            'amssymb',
            'graphicx',
            'xcolor',
            'booktabs',
            'array',
            'longtable',
            'float',
            'caption',
            'subcaption',
            'hyperref',
            'url',
            'listings',
            'algorithm2e',
            'algorithmicx',
            'algpseudocode'
        ]
        
        success_count = 0
        for package in packages:
            if run_command(f'tlmgr install {package}', f'Installing LaTeX package {package}'):
                success_count += 1
        
        print(f"📊 Additional LaTeX packages: {success_count}/{len(packages)} installed successfully")
        return success_count > 0
        
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("⚠️  tlmgr not found - skipping additional package installation")
        print("   You may need to install packages manually later")
        return True

def main():
    print("📄 LaTeX Installation Helper for Causality-Copilot")
    print("=" * 60)
    
    # Check if already installed
    if check_latex():
        print("✅ LaTeX already installed!")
        print("\n🐍 Installing Python LaTeX integration packages...")
        install_python_latex_packages()
        install_additional_latex_packages()
        return
    
    print("\n📋 LaTeX Installation Options:")
    print("1. Install full LaTeX distribution (recommended for full features)")
    print("2. Install TinyTeX (lightweight, good for basic needs)")
    print("3. Skip LaTeX installation (install Python packages only)")
    
    choice = input("Choose option (1, 2, or 3): ").strip()
    
    success = False
    
    if choice == "1":
        # Install full distribution
        system = platform.system()
        if system == "Darwin":
            success = install_latex_macos()
        elif system == "Linux":
            success = install_latex_linux()
        elif system == "Windows":
            success = install_latex_windows()
        else:
            print(f"❌ Unsupported platform: {system}")
    
    elif choice == "2":
        # Install TinyTeX
        success = install_tinytex()
    
    elif choice == "3":
        # Skip LaTeX installation
        success = True
        print("⏭️  Skipping LaTeX distribution installation")
    
    else:
        print("❌ Invalid choice")
        return
    
    # Install Python packages
    print("\n🐍 Installing Python LaTeX integration packages...")
    python_success = install_python_latex_packages()
    
    # Install additional LaTeX packages if LaTeX is available
    if success and choice in ["1", "2"]:
        print("\n📚 Installing additional LaTeX packages...")
        install_additional_latex_packages()
    
    print("\n" + "=" * 60)
    
    if success:
        print("🎉 LaTeX setup completed!")
        print("\n💡 Next steps:")
        print("   1. Restart your terminal to update PATH")
        print("   2. Test LaTeX: pdflatex --version")
        print("   3. Test Python integration: python -c \"import pylatex; print('✅ PyLaTeX ready')\"")
        
        if choice == "2":  # TinyTeX
            print("   4. For TinyTeX, you may need to install additional packages as needed:")
            print("      tlmgr install <package-name>")
    else:
        print("❌ LaTeX installation failed")
        print("   Please install LaTeX manually and then run Python package installation")

if __name__ == "__main__":
    main()
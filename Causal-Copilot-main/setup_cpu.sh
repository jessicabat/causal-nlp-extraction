#!/bin/bash
echo "🚀 Causality-Copilot CPU Setup Script"
echo "====================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install system packages on macOS
install_macos_deps() {
    echo "🍎 Installing macOS dependencies..."
    
    # Check for Homebrew
    if command_exists brew; then
        echo "✅ Homebrew found"
        
        # Install Graphviz
        if ! command_exists dot; then
            echo "📦 Installing Graphviz..."
            brew install graphviz
        else
            echo "✅ Graphviz already installed"
        fi
        
        # Install LaTeX (BasicTeX - lightweight)
        if ! command_exists pdflatex; then
            echo "📄 Installing BasicTeX..."
            brew install --cask basictex
            echo "⚠️  Please restart your terminal after installation completes"
            echo "⚠️  Then add to PATH: export PATH=\"\$PATH:/Library/TeX/texbin\""
        else
            echo "✅ LaTeX already installed"
        fi
    else
        echo "❌ Homebrew not found. Please install Homebrew first:"
        echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi
}

# Function to install system packages on Linux
install_linux_deps() {
    echo "🐧 Installing Linux dependencies..."
    
    # Detect Linux distribution
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$NAME
    else
        echo "❌ Cannot detect Linux distribution"
        exit 1
    fi
    
    case $OS in
        *"Ubuntu"*|*"Debian"*)
            echo "🔵 Detected Ubuntu/Debian"
            sudo apt-get update
            
            # Install Graphviz
            if ! command_exists dot; then
                echo "📦 Installing Graphviz..."
                sudo apt-get install -y graphviz graphviz-dev
            fi
            
            # Install LaTeX
            if ! command_exists pdflatex; then
                echo "📄 Installing TeX Live..."
                sudo apt-get install -y texlive-latex-base texlive-latex-extra texlive-fonts-recommended
            fi
            ;;
        *"CentOS"*|*"Red Hat"*|*"Amazon Linux"*)
            echo "🔴 Detected CentOS/RHEL/Amazon Linux"
            
            # Install Graphviz
            if ! command_exists dot; then
                echo "📦 Installing Graphviz..."
                if command_exists dnf; then
                    sudo dnf install -y graphviz graphviz-devel
                else
                    sudo yum install -y graphviz graphviz-devel
                fi
            fi
            
            # Install LaTeX
            if ! command_exists pdflatex; then
                echo "📄 Installing TeX Live..."
                if command_exists dnf; then
                    sudo dnf install -y texlive-latex texlive-latex-extra
                else
                    sudo yum install -y texlive-latex texlive-latex-extra
                fi
            fi
            ;;
        *)
            echo "⚠️  Unsupported Linux distribution: $OS"
            echo "   Please install graphviz and texlive manually"
            ;;
    esac
}

# Detect operating system and install system dependencies
echo "🔧 Installing system dependencies..."
case "$(uname -s)" in
    Darwin*)
        install_macos_deps
        ;;
    Linux*)
        install_linux_deps
        ;;
    *)
        echo "❌ Unsupported operating system: $(uname -s)"
        echo "   Please install graphviz and LaTeX manually"
        ;;
esac

echo ""
echo "🐍 Installing Python packages..."
echo "================================"

# Install the base packages
echo "📦 Installing base requirements..."
pip install -r requirements_cpu.txt --no-deps --no-cache-dir

# Install the rest of the packages torch needs
echo "🔥 Installing PyTorch..."
pip install torch

# Install the rest of the packages gradio needs  
echo "🌐 Installing Gradio..."
pip install gradio

# Install LaTeX-related Python packages
echo "📄 Installing LaTeX integration packages..."
pip install -r requirements_latex.txt

# Update LaTeX package manager and install essential packages
if command_exists tlmgr; then
    echo "📚 Updating LaTeX packages..."
    tlmgr update --self
    echo "📦 Installing essential LaTeX packages..."
    tlmgr install fancyhdr geometry amsmath booktabs array longtable float caption hyperref url listings xcolor 2>/dev/null || echo "⚠️  Some LaTeX packages may already be installed"
else
    echo "⚠️  tlmgr not found - LaTeX packages will be installed on-demand"
fi

echo ""
echo "✅ Installation completed!"
echo "========================="

echo ""
echo "📄 Running LaTeX functionality test..."
python test_latex_functionality.py

echo ""
echo "🎉 Setup complete! You can now run:"
echo "   python web_demo/demo.py"
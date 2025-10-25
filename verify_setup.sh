#!/bin/bash
# Quick verification script for FieldGuide RAG+LLM setup

echo "🔍 Verifying FieldGuide RAG+LLM Integration..."
echo ""

# Check project structure
echo "📁 Checking project structure..."
if [ -f "app/ios/Runner/LLMService.swift" ]; then
    echo "  ✅ LLMService.swift exists"
else
    echo "  ❌ LLMService.swift missing"
fi

if [ -f "app/ios/Runner/EmbedService.swift" ]; then
    echo "  ✅ EmbedService.swift exists"
else
    echo "  ❌ EmbedService.swift missing"
fi

if [ -f "app/ios/Runner/RAGService.swift" ]; then
    echo "  ✅ RAGService.swift exists"
else
    echo "  ❌ RAGService.swift missing"
fi

if [ -f "app/ios/Podfile" ]; then
    echo "  ✅ Podfile exists"
else
    echo "  ❌ Podfile missing"
fi

if [ -f "rag_database.db" ]; then
    echo "  ✅ RAG database exists"
else
    echo "  ❌ RAG database missing"
fi

echo ""
echo "📝 Checking Flutter files..."
if [ -f "app/lib/services/native_channels.dart" ]; then
    echo "  ✅ native_channels.dart exists"
else
    echo "  ❌ native_channels.dart missing"
fi

if [ -f "app/lib/main.dart" ]; then
    echo "  ✅ main.dart exists"
else
    echo "  ❌ main.dart missing"
fi

echo ""
echo "🔧 Next steps:"
echo "  1. cd app/ios && pod install"
echo "  2. cd .. && flutter pub get"
echo "  3. open ios/Runner.xcworkspace"
echo "  4. flutter run"
echo ""
echo "📖 See MLC_INTEGRATION_GUIDE.md for detailed setup instructions"

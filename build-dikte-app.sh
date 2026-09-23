#!/bin/bash
# Build Dikte.app — a menu bar app that starts and stops Kreyòl dictation, with no terminal open.
#
#   ./build-dikte-app.sh                      # into ~/Applications
#   ./build-dikte-app.sh /Applications        # somewhere else
#
# The app is a thin launcher around dikte/dikte.py in this folder, not a copy of it, so editing the
# script changes the app with no rebuild. Both the interpreter and this folder are written in as
# absolute paths: an app launched from the Finder gets a minimal PATH, none of your shell's setup,
# and no pyenv shims, so "python3" there is not the python3 you have been using.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
INTO="${1:-$HOME/Applications}"
APP="$INTO/Dikte.app"
PYTHON="$(python3 -c 'import sys; print(sys.executable)')"

"$PYTHON" - <<'CHECK' || exit 1
import importlib.util
import sys
needed = ['sounddevice', 'numpy', 'pynput', 'rumps', 'Quartz', 'AppKit', 'ApplicationServices']
missing = [name for name in needed if importlib.util.find_spec(name) is None]
if missing:
    print(f'{sys.executable}\n  is missing: ' + ', '.join(missing), file=sys.stderr)
    print('  install them into that python, not another one:', file=sys.stderr)
    print('  python3 -m pip install sounddevice pynput rumps pyobjc-framework-Quartz '
          'pyobjc-framework-Cocoa', file=sys.stderr)
    raise SystemExit(1)
print(f'using {sys.executable}')
CHECK

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Dikte</string>
  <key>CFBundleDisplayName</key><string>Dikte</string>
  <key>CFBundleIdentifier</key><string>ht.kreyol.dikte</string>
  <key>CFBundleExecutable</key><string>Dikte</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleVersion</key><string>1.0</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <!-- Menu bar only: no Dock icon, no window. -->
  <key>LSUIElement</key><true/>
  <!-- macOS shows this sentence when it asks for the microphone. -->
  <key>NSMicrophoneUsageDescription</key>
  <string>Dikte records your voice so the Kreyòl speech model on this Mac can write down what you said.</string>
</dict>
</plist>
PLIST

cat > "$APP/Contents/MacOS/Dikte" <<LAUNCHER
#!/bin/bash
# Written by build-dikte-app.sh. Absolute paths on purpose — see that script for why.
LOG="\$HOME/Library/Logs/Dikte.log"
mkdir -p "\$(dirname "\$LOG")"
echo "--- started \$(date) ---" >> "\$LOG"
exec "$PYTHON" "$DIR/dikte/dikte.py" "\$@" >> "\$LOG" 2>&1
LAUNCHER
chmod +x "$APP/Contents/MacOS/Dikte"

# Ad-hoc signature. macOS ties Accessibility to an app's identity; without one, every rebuild can look
# like a different app and you would have to grant the permission again.
if command -v codesign > /dev/null; then
  codesign --force --sign - "$APP" 2> /dev/null && echo "signed (ad-hoc)" || echo "could not sign; permissions may need re-granting after a rebuild"
fi

echo "built $APP"
echo
echo "Next: open it once, then grant it Accessibility —"
echo "  open \"$APP\""
echo "  open \"x-apple.systempreferences:com.apple.settings.PrivacySecurity.extension?Privacy_Accessibility\""
echo "Add Dikte, switch it on, then quit Dikte from its menu and open it again."
echo "Its log: ~/Library/Logs/Dikte.log"

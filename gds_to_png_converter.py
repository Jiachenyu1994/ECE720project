import pya
import os
import time

# Get parameters
try:
    IN_FILE = "C:\\Users\\steph\\Desktop\\UWaterloo\\spring2025\\ECE720-ML for chip\\git repo\\ECE720project\\gds_files\\sky130_fd_sc_hd__a2bb2o_1.gds"
    print(f"✓ Input GDS: {IN_FILE}")
except NameError:
    print("Error: No input GDS specified")
    exit(1)

try:
    OUT_FILE = "C:\\Users\\steph\\Desktop\\UWaterloo\\spring2025\\ECE720-ML for chip\\git repo\\ECE720project\\output_images\\sky130_fd_sc_hd__a2bb2o_1.png"
except NameError:
    OUT_FILE = IN_FILE.replace('.gds', '.png')

try:
    WIDTH = int(width)
except NameError:
    WIDTH = 1200

try:
    HEIGHT = int(height)
except NameError:
    HEIGHT = 800

print(f"✓ Output PNG: {OUT_FILE}")
print(f"✓ Dimensions: {WIDTH}x{HEIGHT}")

def analyze_cell_content():
    """Just analyze what's in the cell without complex API calls"""
    try:
        layout = pya.Layout()
        layout.read(IN_FILE)

        if layout.cells() == 0:
            print("   ✗ No cells in layout")
            return False

        cell = layout.cell(0)
        bbox = cell.bbox()

        print(f"\n📋 Cell Analysis:")
        print(f"   📦 Cell name: {cell.name}")
        print(f"   📏 Bounding box: {bbox}")
        print(f"   📐 Width: {bbox.width()} DBU")
        print(f"   📐 Height: {bbox.height()} DBU")
        print(f"   🔍 Empty: {bbox.empty()}")

        if bbox.empty():
            print("   ⚠ WARNING: Cell appears to be empty!")
            return False

        # Count layers with content
        layers_with_content = 0
        for layer_idx in range(layout.layers()):
            layer_info = layout.get_info(layer_idx)
            if layer_info and not cell.shapes(layer_idx).is_empty():
                layers_with_content += 1

        print(f"   🎨 Layers with content: {layers_with_content}")

        if layers_with_content == 0:
            print("   ⚠ WARNING: No layers appear to have content!")
            return False

        return True

    except Exception as e:
        print(f"   ✗ Analysis failed: {e}")
        return False

def simple_png_export():
    """Simplest possible PNG export - just load and export"""

    print(f"\n📸 Simple PNG Export (No Layer Manipulation)")
    print(f"🔄 Converting: {os.path.basename(IN_FILE)}")

    try:
        # Check if we're in GUI mode
        app = pya.Application.instance()
        if not app:
            print("   ✗ No KLayout application (need GUI mode - remove -b flag)")
            return False

        mw = app.main_window()
        if not mw:
            print("   ✗ No main window (need GUI mode - remove -b flag)")
            return False

        print("   📂 Loading GDS file...")

        # Load the layout - simple approach
        mw.load_layout(IN_FILE, 0)

        # Get the current view
        lv = mw.current_view()
        if not lv:
            print("   ✗ No current view available")
            return False

        print("   ⏱️ Waiting for layout to load...")
        time.sleep(1)  # Give it a moment to load

        # Try to get some info about what's loaded
        try:
            cellview = lv.cellview(0)
            if cellview.is_valid():
                layout = cellview.layout()
                if layout and layout.cells() > 0:
                    cell = layout.cell(0)
                    print(f"   📦 Loaded cell: {cell.name}")
                    print(f"   📏 Bbox: {cell.bbox()}")
        except:
            print("   📦 Layout loaded (couldn't get details)")

        # Zoom to fit
        print("   🔍 Zooming to fit...")
        lv.zoom_fit()

        # Small delay to let zoom complete
        time.sleep(0.5)

        # Export PNG - your original simple method
        print(f"   💾 Exporting PNG ({WIDTH}x{HEIGHT})...")
        lv.save_image(OUT_FILE, WIDTH, HEIGHT)

        # Check if file was created
        if os.path.exists(OUT_FILE):
            file_size = os.path.getsize(OUT_FILE)
            print(f"   ✓ PNG file created!")
            print(f"   📊 Size: {file_size} bytes")

            if file_size < 5000:
                print("   ⚠ File is quite small - may indicate:")
                print("      - Layers not visible by default")
                print("      - Cell is very small")
                print("      - Need to adjust layer visibility manually")
            else:
                print("   🎉 Good file size - likely has visible content!")

            return True
        else:
            print("   ✗ PNG file was not created")
            return False

    except Exception as e:
        print(f"   ✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("📸 Simple PNG Export")
    print("=" * 50)

    # First analyze the cell
    if not analyze_cell_content():
        print("\n❌ Cell analysis failed - check your GDS file")
        return

    # Then try to export
    success = simple_png_export()

    print("=" * 50)
    if success:
        print("✅ Export completed!")
        print(f"📸 PNG file: {os.path.abspath(OUT_FILE)}")
        print(f"\n💡 Next steps:")
        print(f"   1. Check the PNG file to see if layers are visible")
        print(f"   2. If empty/white, try manual layer setup:")
        print(f"      klayout {IN_FILE}")
        print(f"      → Check layer panel and make layers visible")
        print(f"      → File → Export → PNG")
    else:
        print("❌ Export failed")
        print(f"\n🔧 Manual approach (guaranteed to work):")
        print(f"   1. klayout {IN_FILE}")
        print(f"   2. Look at the layer panel (usually on right side)")
        print(f"   3. Check/uncheck layers to make them visible")
        print(f"   4. File → Export → PNG → {WIDTH}x{HEIGHT}")

if __name__ == "__main__":
    main()

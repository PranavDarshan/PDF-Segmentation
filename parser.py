import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import os
from pathlib import Path
import json


class AnswerScriptParser:
    """
    Robust parser for answer scripts with accurate question number detection
    """
    
    def __init__(self, output_dir="output", margin_width_ratio=0.08):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.margin_width_ratio = margin_width_ratio
        
        (self.output_dir / "questions").mkdir(exist_ok=True)
        (self.output_dir / "pages").mkdir(exist_ok=True)
        (self.output_dir / "visualization").mkdir(exist_ok=True)
        (self.output_dir / "margins").mkdir(exist_ok=True)
        
    def pdf_to_images(self, pdf_path, dpi=300):
        """Convert PDF to images"""
        print(f"Converting PDF to images (DPI: {dpi})...")
        images = convert_from_path(pdf_path, dpi=dpi)
        
        for idx, img in enumerate(images):
            img.save(self.output_dir / "pages" / f"page_{idx+1}.png")
        
        print(f"✓ Extracted {len(images)} pages")
        return images
    
    def extract_margin(self, img):
        """Extract left margin region"""
        height, width = img.shape[:2]
        margin_width = int(width * self.margin_width_ratio)
        
        margin = img[:, :margin_width]
        content = img[:, margin_width:]
        
        return margin, content, margin_width
    
    def detect_question_numbers_robust(self, margin_img, page_num):
        """
        Robust question number detection for handwritten numbers
        """
        if len(margin_img.shape) == 3:
            gray = cv2.cvtColor(margin_img, cv2.COLOR_RGB2GRAY)
        else:
            gray = margin_img
        
        # Adaptive thresholding with larger block size
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 8
        )
        
        # Remove small noise
        kernel_noise = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_noise)
        
        # Dilate slightly to connect number parts
        kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.dilate(binary, kernel_connect, iterations=1)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        
        # Create debug visualization
        margin_debug = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Draw all components with their areas
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Draw all components in red
            cv2.rectangle(margin_debug, (x, y), (x+w, y+h), (0, 0, 255), 1)
            cv2.putText(margin_debug, f"{area}", (x, y-2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        
        # Save debug images
        cv2.imwrite(
            str(self.output_dir / "margins" / f"page_{page_num}_all_components.png"),
            margin_debug
        )
        cv2.imwrite(
            str(self.output_dir / "margins" / f"page_{page_num}_margin_gray.png"),
            gray
        )
        cv2.imwrite(
            str(self.output_dir / "margins" / f"page_{page_num}_binary.png"),
            binary
        )
        
        candidates = []
        
        # Filter for handwritten question numbers
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            
            aspect_ratio = w / h if h > 0 else 0
            density = area / (w * h) if (w * h) > 0 else 0
            
            # FILTERING for handwritten question numbers
            # Area: 500-8000 pixels for handwritten numbers
            if not (500 < area < 8000):
                continue
            
            # Aspect ratio: reasonable shape
            if not (0.2 < aspect_ratio < 4.0):
                continue
            
            # Density: solid writing
            if density < 0.15:
                continue
            
            # Width check: not too wide
            if w > margin_img.shape[1] * 0.8:
                continue
            
            # Height check: realistic for numbers
            if h < 20 or h > 200:
                continue
            
            y_center = int(centroids[i][1])
            
            candidates.append({
                'y_position': y_center,
                'page': page_num,
                'bbox': (x, y, w, h),
                'area': area,
                'aspect_ratio': aspect_ratio,
                'density': density
            })
        
        # Sort by y position
        candidates.sort(key=lambda m: m['y_position'])
        
        # Remove duplicates - keep larger ones
        filtered = []
        min_vertical_distance = 200
        
        for candidate in candidates:
            if not filtered:
                filtered.append(candidate)
            else:
                too_close = False
                for idx, existing in enumerate(filtered):
                    if abs(candidate['y_position'] - existing['y_position']) < min_vertical_distance:
                        # Keep the larger one
                        if candidate['area'] > existing['area']:
                            filtered[idx] = candidate
                        too_close = True
                        break
                
                if not too_close:
                    filtered.append(candidate)
        
        # Sort again
        filtered.sort(key=lambda m: m['y_position'])
        
        # Draw filtered candidates in green on debug image
        for marker in filtered:
            x, y, w, h = marker['bbox']
            cv2.rectangle(margin_debug, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(margin_debug, f"Q{filtered.index(marker)+1}", 
                       (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save final debug image
        cv2.imwrite(
            str(self.output_dir / "margins" / f"page_{page_num}_detected.png"),
            margin_debug
        )
        
        return filtered
    
    def build_answer_spans(self, all_page_data):
        """Build answer spans across multiple pages"""
        answer_spans = []
        current_span = None
        
        for page_data in all_page_data:
            page_num = page_data['page_num']
            markers = page_data['markers']
            height = page_data['height']
            
            if not markers:
                if current_span:
                    current_span['pages'].append({
                        'page': page_num,
                        'y_start': 0,
                        'y_end': height
                    })
                else:
                    current_span = {
                        'question_number': 1,
                        'pages': [{
                            'page': page_num,
                            'y_start': 0,
                            'y_end': height
                        }]
                    }
            else:
                for idx, marker in enumerate(markers):
                    if current_span:
                        current_span['pages'].append({
                            'page': page_num,
                            'y_start': 0,
                            'y_end': marker['y_position']
                        })
                        answer_spans.append(current_span)
                    
                    current_question = len(answer_spans) + 1
                    
                    if idx < len(markers) - 1:
                        y_end = markers[idx + 1]['y_position']
                    else:
                        y_end = height
                    
                    current_span = {
                        'question_number': current_question,
                        'pages': [{
                            'page': page_num,
                            'y_start': max(0, marker['y_position'] - 20),
                            'y_end': y_end
                        }]
                    }
        
        if current_span:
            answer_spans.append(current_span)
        
        return answer_spans
    
    def detect_regions_in_image(self, img, min_area=1000):
        """Detect text and diagram regions"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        dilated = cv2.dilate(binary, kernel, iterations=2)
        
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            
            regions.append({
                "bbox": (x, y, w, h),
                "area": area,
                "aspect_ratio": w / h if h > 0 else 0,
                "width": w,
                "height": h
            })
        
        regions.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))
        return regions
    
    def classify_region(self, region_img, region_info):
        """Classify region as text or diagram"""
        if len(region_img.shape) == 3:
            gray = cv2.cvtColor(region_img, cv2.COLOR_RGB2GRAY)
        else:
            gray = region_img
        
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, threshold=30,
            minLineLength=30, maxLineGap=10
        )
        
        aspect_ratio = region_info["aspect_ratio"]
        is_diagram = False
        
        if lines is not None:
            num_lines = len(lines)
            horizontal = vertical = diagonal = 0
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                
                if angle < 20 or angle > 160:
                    horizontal += 1
                elif 70 < angle < 110:
                    vertical += 1
                else:
                    diagonal += 1
            
            if vertical > 4 or diagonal > 3:
                is_diagram = True
            
            if num_lines > 15 and 0.5 < aspect_ratio < 2.0:
                is_diagram = True
        
        if edge_density > 0.2 and 0.4 < aspect_ratio < 2.5:
            is_diagram = True
        
        if aspect_ratio > 5:
            is_diagram = False
        
        return "diagram" if is_diagram else "text"
    
    def visualize_detection(self, pages_imgs, all_page_data, answer_spans):
        """Create visualization"""
        for page_data in all_page_data:
            page_num = page_data['page_num']
            img = pages_imgs[page_num - 1]
            markers = page_data['markers']
            margin_width = page_data['margin_width']
            
            vis = img.copy()
            
            # Draw margin line
            cv2.line(vis, (margin_width, 0), (margin_width, img.shape[0]), 
                     (255, 0, 0), 3)
            
            # Draw question markers
            for marker in markers:
                y = marker['y_position']
                x, my, w, h = marker['bbox']
                
                # Draw marker bounding box in yellow
                cv2.rectangle(vis, (x, my), (x+w, my+h), (0, 255, 255), 3)
                
                # Draw horizontal line
                cv2.line(vis, (0, y), (img.shape[1], y), (0, 255, 0), 3)
            
            # Draw answer spans
            for answer in answer_spans:
                for page_span in answer['pages']:
                    if page_span['page'] == page_num:
                        q_num = answer['question_number']
                        y_start = page_span['y_start']
                        y_end = page_span['y_end']
                        
                        cv2.rectangle(
                            vis, (margin_width, y_start), (img.shape[1], y_end),
                            (255, 0, 255), 4
                        )
                        
                        cv2.putText(
                            vis, f"Q{q_num}", (margin_width + 30, y_start + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4
                        )
            
            vis_path = self.output_dir / "visualization" / f"page_{page_num}_detection.png"
            cv2.imwrite(str(vis_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    
    def extract_answer_content(self, pages_imgs, answer_span):
        """Extract answer content (possibly multi-page)"""
        combined_parts = []
        
        for page_span in answer_span['pages']:
            page_num = page_span['page']
            y_start = max(0, page_span['y_start'])
            y_end = page_span['y_end']
            
            page_img = pages_imgs[page_num - 1]
            section = page_img[y_start:y_end, :]
            combined_parts.append(section)
        
        if len(combined_parts) == 1:
            combined_img = combined_parts[0]
        else:
            combined_img = np.vstack(combined_parts)
        
        return combined_img
    
    def process_answer(self, pages_imgs, answer_span):
        """Process a single answer"""
        q_num = answer_span['question_number']
        
        print(f"\n  Q{q_num}:", end=" ")
        
        page_nums = [p['page'] for p in answer_span['pages']]
        if len(page_nums) == 1:
            print(f"Page {page_nums[0]}")
        else:
            print(f"Pages {page_nums[0]}-{page_nums[-1]}")
        
        answer_img = self.extract_answer_content(pages_imgs, answer_span)
        
        q_dir = self.output_dir / "questions" / f"Q{q_num}"
        q_dir.mkdir(exist_ok=True)
        
        cv2.imwrite(
            str(q_dir / "full_answer.png"),
            cv2.cvtColor(answer_img, cv2.COLOR_RGB2BGR)
        )
        
        regions = self.detect_regions_in_image(answer_img)
        
        text_count = 0
        diagram_count = 0
        
        answer_data = {
            "question_number": q_num,
            "spans_pages": page_nums,
            "page_spans": answer_span['pages'],
            "text_regions": [],
            "diagrams": []
        }
        
        for region in regions:
            x, y, w, h = region["bbox"]
            region_img = answer_img[y:y+h, x:x+w]
            
            region_type = self.classify_region(region_img, region)
            
            if region_type == "text":
                text_count += 1
                filename = f"text_{text_count}.png"
                answer_data["text_regions"].append({
                    "filename": filename,
                    "bbox": [x, y, w, h]
                })
            else:
                diagram_count += 1
                filename = f"diagram_{diagram_count}.png"
                answer_data["diagrams"].append({
                    "filename": filename,
                    "bbox": [x, y, w, h]
                })
            
            cv2.imwrite(
                str(q_dir / filename),
                cv2.cvtColor(region_img, cv2.COLOR_RGB2BGR)
            )
        
        with open(q_dir / "metadata.json", 'w') as f:
            json.dump(answer_data, f, indent=2)
        
        print(f"    → {text_count} text, {diagram_count} diagrams")
        
        return answer_data
    
    def process_pdf(self, pdf_path, dpi=300, margin_width_ratio=0.08):
        """Main processing pipeline"""
        print("\n" + "="*60)
        print("ANSWER SCRIPT PARSER - HANDWRITTEN Q NUMBERS")
        print("="*60)
        
        self.margin_width_ratio = margin_width_ratio
        
        pages = self.pdf_to_images(pdf_path, dpi)
        pages_imgs = [np.array(p) for p in pages]
        
        print("\n📋 Detecting question numbers...")
        
        all_page_data = []
        for page_num, img in enumerate(pages_imgs, 1):
            margin, content, margin_width = self.extract_margin(img)
            markers = self.detect_question_numbers_robust(margin, page_num)
            
            print(f"  Page {page_num}: {len(markers)} question(s) detected")
            
            if markers:
                print(f"    Y-positions: {[m['y_position'] for m in markers]}")
                print(f"    Areas: {[m['area'] for m in markers]}")
            
            all_page_data.append({
                'page_num': page_num,
                'markers': markers,
                'margin_width': margin_width,
                'height': img.shape[0]
            })
        
        print("\n🔗 Building answer spans...")
        answer_spans = self.build_answer_spans(all_page_data)
        
        print(f"\n✓ Total questions: {len(answer_spans)}")
        
        print("\n🎨 Creating visualizations...")
        self.visualize_detection(pages_imgs, all_page_data, answer_spans)
        
        print("\n⚙️  Processing answers...")
        
        results = {
            "pdf_name": Path(pdf_path).name,
            "total_pages": len(pages),
            "total_questions": len(answer_spans),
            "questions": []
        }
        
        for answer_span in answer_spans:
            answer_data = self.process_answer(pages_imgs, answer_span)
            results["questions"].append(answer_data)
        
        with open(self.output_dir / "summary.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        self.print_summary(results)
        
        return results
    
    def print_summary(self, results):
        """Print summary"""
        print("\n" + "="*60)
        print("PROCESSING COMPLETE ✓")
        print("="*60)
        print(f"\nPDF: {results['pdf_name']}")
        print(f"Pages: {results['total_pages']}")
        print(f"Questions Detected: {results['total_questions']}")
        
        multi_page = [q for q in results['questions'] if len(q['spans_pages']) > 1]
        if multi_page:
            print(f"\nMulti-page answers: {len(multi_page)}")
            for q in multi_page:
                pages = q['spans_pages']
                print(f"  Q{q['question_number']}: Pages {pages[0]}-{pages[-1]}")
        
        total_text = sum(len(q['text_regions']) for q in results['questions'])
        total_diagrams = sum(len(q['diagrams']) for q in results['questions'])
        
        print(f"\nTotal Text Regions: {total_text}")
        print(f"Total Diagrams: {total_diagrams}")
        
        print(f"\n📁 Output: {self.output_dir.absolute()}")
        print("\n⚠️  CHECK THESE FOLDERS:")
        print("  1. margins/ - See detected components with areas")
        print("  2. visualization/ - See final detection on pages")
        print("\nIf detection is wrong, check component areas in margins/")
        print("and adjust area range in code (line ~128)")


if __name__ == "__main__":
    parser = AnswerScriptParser(output_dir="parsed_output")
    
    pdf_path = "answer_script.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"\n❌ PDF not found: {pdf_path}")
    else:
        results = parser.process_pdf(
            pdf_path=pdf_path,
            dpi=300,
            margin_width_ratio=0.08  # Adjust: 0.05-0.15
        )
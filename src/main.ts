import "./style.css";
import { SelectionManager } from "./ui-utils.js";
import { EvaluationManager } from "./evaluation-manager.js";

export interface Point {
  x: number;
  y: number;
}

export interface DetectedShape {
  type: "circle" | "triangle" | "rectangle" | "pentagon" | "star";
  confidence: number;
  boundingBox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  center: Point;
  area: number;
}

export interface DetectionResult {
  shapes: DetectedShape[];
  processingTime: number;
  imageWidth: number;
  imageHeight: number;
}

interface BlobData {
  pixels: Point[];
  perimeterPixels: Point[];
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

export class ShapeDetector {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d")!;
  }

  async detectShapes(imageData: ImageData): Promise<DetectionResult> {
    const startTime = performance.now();

    const { width, height, data } = imageData;
    const visited = new Uint8Array(width * height);
    const shapes: DetectedShape[] = [];

    // Block transparent noise tiles (low alpha) and light background
    // Noisy background tiles composite to RGB ~217-242, so threshold at 210
    const isShapePixel = (idx: number) => {
      const r = data[idx],
        g = data[idx + 1],
        b = data[idx + 2],
        a = data[idx + 3];
      const isBackground = r > 210 && g > 210 && b > 210;
      return a > 100 && !isBackground;
    };

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const pixelIndex = y * width + x;

        if (!visited[pixelIndex] && isShapePixel(pixelIndex * 4)) {
          const blob = this.extractBlob(
            x,
            y,
            width,
            height,
            visited,
            isShapePixel
          );

          const boxWidth = blob.maxX - blob.minX + 1;
          const boxHeight = blob.maxY - blob.minY + 1;
          const area = blob.pixels.length;
          const perimeter = blob.perimeterPixels.length;

          // Solidity check: reject thin lines/text (high perimeter vs area)
          const isSolid = area > perimeter * 1.5;

          if (area > 200 && boxWidth > 15 && boxHeight > 15 && isSolid) {
            const shape = this.classifyShape(blob, boxWidth, boxHeight);
            if (shape) shapes.push(shape);
          }
        }
      }
    }

    const processingTime = performance.now() - startTime;

    return {
      shapes,
      processingTime,
      imageWidth: imageData.width,
      imageHeight: imageData.height,
    };
  }

  private extractBlob(
    startX: number,
    startY: number,
    width: number,
    height: number,
    visited: Uint8Array,
    isShapePixel: (idx: number) => boolean
  ): BlobData {
    const queue = [{ x: startX, y: startY }];
    visited[startY * width + startX] = 1;

    const blob: BlobData = {
      pixels: [],
      perimeterPixels: [],
      minX: startX,
      maxX: startX,
      minY: startY,
      maxY: startY,
    };

    let head = 0;
    while (head < queue.length) {
      const { x, y } = queue[head++];
      blob.pixels.push({ x, y });

      if (x < blob.minX) blob.minX = x;
      if (x > blob.maxX) blob.maxX = x;
      if (y < blob.minY) blob.minY = y;
      if (y > blob.maxY) blob.maxY = y;

      const neighbors = [
        { dx: 0, dy: -1 },
        { dx: 0, dy: 1 },
        { dx: -1, dy: 0 },
        { dx: 1, dy: 0 },
      ];
      let isEdge = false;

      for (const { dx, dy } of neighbors) {
        const nx = x + dx,
          ny = y + dy;
        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
          const nIdx = ny * width + nx;
          if (isShapePixel(nIdx * 4)) {
            if (!visited[nIdx]) {
              visited[nIdx] = 1;
              queue.push({ x: nx, y: ny });
            }
          } else {
            isEdge = true;
          }
        } else {
          isEdge = true;
        }
      }
      if (isEdge) blob.perimeterPixels.push({ x, y });
    }
    return blob;
  }

  private countVertices(
    perimeterPixels: Point[],
    centerX: number,
    centerY: number
  ): number {
    if (perimeterPixels.length === 0) return 0;

    // Build radial distance profile over 360 degrees
    const profile = new Array(360).fill(0);
    for (const p of perimeterPixels) {
      const dx = p.x - centerX;
      const dy = p.y - centerY;
      const angle =
        Math.floor((Math.atan2(dy, dx) * 180) / Math.PI + 360) % 360;
      const dist = Math.sqrt(dx * dx + dy * dy);
      profile[angle] = Math.max(profile[angle], dist);
    }

    // Forward-fill empty buckets
    for (let i = 0; i < 360; i++) {
      if (profile[i] === 0) profile[i] = profile[(i - 1 + 360) % 360];
    }

    // Smooth with sliding window to remove pixel noise
    const smoothed = new Array(360).fill(0);
    const halfWin = 7;
    const winSize = halfWin * 2 + 1;
    for (let i = 0; i < 360; i++) {
      let sum = 0;
      for (let j = -halfWin; j <= halfWin; j++) {
        sum += profile[(i + j + 360) % 360];
      }
      smoothed[i] = sum / winSize;
    }

    // Count local maxima (peaks) with ±15° neighborhood
    let peaks = 0;
    const peakAngles: number[] = [];

    for (let i = 0; i < 360; i++) {
      const current = smoothed[i];
      let isPeak = true;

      for (let j = -15; j <= 15; j++) {
        if (j === 0) continue;
        if (smoothed[(i + j + 360) % 360] > current) {
          isPeak = false;
          break;
        }
      }

      if (isPeak) {
        const lastAngle = peakAngles[peakAngles.length - 1];
        if (
          lastAngle === undefined ||
          Math.min(Math.abs(i - lastAngle), 360 - Math.abs(i - lastAngle)) > 20
        ) {
          peaks++;
          peakAngles.push(i);
        }
      }
    }
    return peaks;
  }

  private classifyShape(
    blob: BlobData,
    boxWidth: number,
    boxHeight: number
  ): DetectedShape | null {
    const area = blob.pixels.length;
    const perimeter = blob.perimeterPixels.length;

    let sumX = 0,
      sumY = 0;
    for (const p of blob.pixels) {
      sumX += p.x;
      sumY += p.y;
    }
    const centerX = sumX / area;
    const centerY = sumY / area;

    const circularity = (4 * Math.PI * area) / (perimeter * perimeter);
    const vertices = this.countVertices(
      blob.perimeterPixels,
      centerX,
      centerY
    );

    let type: "circle" | "rectangle" | "triangle" | "pentagon" | "star" =
      "circle";
    let confidence = 0.9;

    if (vertices === 3) {
      type = "triangle";
      confidence = 0.92;
    } else if (vertices === 4) {
      type = "rectangle";
      confidence = 0.95;
    } else if (vertices === 5) {
      // Star vs pentagon: stars have very low circularity (concave indentations)
      if (circularity < 0.5) {
        type = "star";
        confidence = 0.88;
      } else {
        type = "pentagon";
        confidence = 0.85;
      }
    } else if (vertices === 10) {
      // 10 peaks = 5 outer + 5 inner tips of a star
      type = "star";
      confidence = 0.9;
    } else {
      // Fallback for circles or noisy vertex counts
      if (circularity >= 0.78) {
        type = "circle";
        confidence = circularity;
      } else if (circularity < 0.4) {
        type = "star";
        confidence = 0.8;
      } else {
        type = "pentagon";
        confidence = 0.6;
      }
    }

    return {
      type,
      center: { x: centerX, y: centerY },
      boundingBox: {
        x: blob.minX,
        y: blob.minY,
        width: boxWidth,
        height: boxHeight,
      },
      area: area,
      confidence: Math.min(Math.max(confidence, 0.1), 1.0),
    };
  }

  loadImage(file: File): Promise<ImageData> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        this.canvas.width = img.width;
        this.canvas.height = img.height;
        this.ctx.drawImage(img, 0, 0);
        const imageData = this.ctx.getImageData(0, 0, img.width, img.height);
        resolve(imageData);
      };
      img.onerror = reject;
      img.src = URL.createObjectURL(file);
    });
  }
}

class ShapeDetectionApp {
  private detector: ShapeDetector;
  private imageInput: HTMLInputElement;
  private resultsDiv: HTMLDivElement;
  private testImagesDiv: HTMLDivElement;
  private evaluateButton: HTMLButtonElement;
  private evaluationResultsDiv: HTMLDivElement;
  private selectionManager: SelectionManager;
  private evaluationManager: EvaluationManager;

  constructor() {
    const canvas = document.getElementById(
      "originalCanvas"
    ) as HTMLCanvasElement;
    this.detector = new ShapeDetector(canvas);

    this.imageInput = document.getElementById("imageInput") as HTMLInputElement;
    this.resultsDiv = document.getElementById("results") as HTMLDivElement;
    this.testImagesDiv = document.getElementById(
      "testImages"
    ) as HTMLDivElement;
    this.evaluateButton = document.getElementById(
      "evaluateButton"
    ) as HTMLButtonElement;
    this.evaluationResultsDiv = document.getElementById(
      "evaluationResults"
    ) as HTMLDivElement;

    this.selectionManager = new SelectionManager();
    this.evaluationManager = new EvaluationManager(
      this.detector,
      this.evaluateButton,
      this.evaluationResultsDiv
    );

    this.setupEventListeners();
    this.loadTestImages().catch(console.error);
  }

  private setupEventListeners(): void {
    this.imageInput.addEventListener("change", async (event) => {
      const file = (event.target as HTMLInputElement).files?.[0];
      if (file) {
        await this.processImage(file);
      }
    });

    this.evaluateButton.addEventListener("click", async () => {
      const selectedImages = this.selectionManager.getSelectedImages();
      await this.evaluationManager.runSelectedEvaluation(selectedImages);
    });
  }

  private async processImage(file: File): Promise<void> {
    try {
      this.resultsDiv.innerHTML = "<p>Processing...</p>";

      const imageData = await this.detector.loadImage(file);
      const results = await this.detector.detectShapes(imageData);

      this.displayResults(results);
    } catch (error) {
      this.resultsDiv.innerHTML = `<p>Error: ${error}</p>`;
    }
  }

  private displayResults(results: DetectionResult): void {
    const { shapes, processingTime } = results;

    let html = `
      <p><strong>Processing Time:</strong> ${processingTime.toFixed(2)}ms</p>
      <p><strong>Shapes Found:</strong> ${shapes.length}</p>
    `;

    if (shapes.length > 0) {
      html += "<h4>Detected Shapes:</h4><ul>";
      shapes.forEach((shape) => {
        html += `
          <li>
            <strong>${
              shape.type.charAt(0).toUpperCase() + shape.type.slice(1)
            }</strong><br>
            Confidence: ${(shape.confidence * 100).toFixed(1)}%<br>
            Center: (${shape.center.x.toFixed(1)}, ${shape.center.y.toFixed(
          1
        )})<br>
            Area: ${shape.area.toFixed(1)}px²
          </li>
        `;
      });
      html += "</ul>";
    } else {
      html +=
        "<p>No shapes detected. Please implement the detection algorithm.</p>";
    }

    this.resultsDiv.innerHTML = html;
  }

  private async loadTestImages(): Promise<void> {
    try {
      const module = await import("./test-images-data.js");
      const testImages = module.testImages;
      const imageNames = module.getAllTestImageNames();

      let html =
        '<h4>Click to upload your own image or use test images for detection. Right-click test images to select/deselect for evaluation:</h4><div class="evaluation-controls"><button id="selectAllBtn">Select All</button><button id="deselectAllBtn">Deselect All</button><span class="selection-info">0 images selected</span></div><div class="test-images-grid">';

      // Add upload functionality as first grid item
      html += `
        <div class="test-image-item upload-item" onclick="triggerFileUpload()">
          <div class="upload-icon">📁</div>
          <div class="upload-text">Upload Image</div>
          <div class="upload-subtext">Click to select file</div>
        </div>
      `;

      imageNames.forEach((imageName) => {
        const dataUrl = testImages[imageName as keyof typeof testImages];
        const displayName = imageName
          .replace(/[_-]/g, " ")
          .replace(/\.(svg|png)$/i, "");
        html += `
          <div class="test-image-item" data-image="${imageName}" 
               onclick="loadTestImage('${imageName}', '${dataUrl}')" 
               oncontextmenu="toggleImageSelection(event, '${imageName}')">
            <img src="${dataUrl}" alt="${imageName}">
            <div>${displayName}</div>
          </div>
        `;
      });

      html += "</div>";
      this.testImagesDiv.innerHTML = html;

      this.selectionManager.setupSelectionControls();

      (window as any).loadTestImage = async (name: string, dataUrl: string) => {
        try {
          const response = await fetch(dataUrl);
          const blob = await response.blob();
          const file = new File([blob], name, { type: "image/svg+xml" });

          const imageData = await this.detector.loadImage(file);
          const results = await this.detector.detectShapes(imageData);
          this.displayResults(results);

          console.log(`Loaded test image: ${name}`);
        } catch (error) {
          console.error("Error loading test image:", error);
        }
      };

      (window as any).toggleImageSelection = (
        event: MouseEvent,
        imageName: string
      ) => {
        event.preventDefault();
        this.selectionManager.toggleImageSelection(imageName);
      };

      // Add upload functionality
      (window as any).triggerFileUpload = () => {
        this.imageInput.click();
      };
    } catch (error) {
      this.testImagesDiv.innerHTML = `
        <p>Test images not available. Run 'node convert-svg-to-png.js' to generate test image data.</p>
        <p>SVG files are available in the test-images/ directory.</p>
      `;
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  new ShapeDetectionApp();
});

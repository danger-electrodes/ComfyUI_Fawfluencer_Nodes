import { app } from "../../../scripts/app.js";


let elementWidgets = null;

const originalHas = Set.prototype.has;
Set.prototype.has = function(value) {
    if (value && app && app.graph && app.graph.nodes.includes(value)) {
      elementWidgets = this; 
        Set.prototype.has = originalHas;
    }
    return originalHas.call(this, value);
};

//DOMWidgetImpl can be replaced by window.comfyAPI.domWidget.DOMWidgetImpl
var __defProp2 = Object.defineProperty;
var __name = (target2, value4) => __defProp2(target2, "name", { value: value4, configurable: true });
const SIZE = Symbol();

//const DOMWidgetImpl = window.comfyAPI.domWidget.DOMWidgetImpl;

function intersect(a2, b2) {
  const x2 = Math.max(a2.x, b2.x);
  const num1 = Math.min(a2.x + a2.width, b2.x + b2.width);
  const y2 = Math.max(a2.y, b2.y);
  const num2 = Math.min(a2.y + a2.height, b2.y + b2.height);
  if (num1 >= x2 && num2 >= y2) return [x2, y2, num1 - x2, num2 - y2];
  else return null;
}
__name(intersect, "intersect");

function getClipPath(node3, element, canvasRect) {
  const selectedNode = Object.values(
    app.canvas.selected_nodes ?? {}
  )[0];
  if (selectedNode && selectedNode !== node3) {
    const elRect = element.getBoundingClientRect();
    const MARGIN = 4;
    const { offset, scale } = app.canvas.ds;
    const { renderArea } = selectedNode;
    const intersection = intersect(
      {
        x: elRect.left - canvasRect.left,
        y: elRect.top - canvasRect.top,
        width: elRect.width,
        height: elRect.height
      },
      {
        x: (renderArea[0] + offset[0] - MARGIN) * scale,
        y: (renderArea[1] + offset[1] - MARGIN) * scale,
        width: (renderArea[2] + 2 * MARGIN) * scale,
        height: (renderArea[3] + 2 * MARGIN) * scale
      }
    );
    if (!intersection) {
      return "";
    }
    const clipX = (intersection[0] - elRect.left + canvasRect.left) / scale + "px";
    const clipY = (intersection[1] - elRect.top + canvasRect.top) / scale + "px";
    const clipWidth = intersection[2] / scale + "px";
    const clipHeight = intersection[3] / scale + "px";
    const path = `polygon(0% 0%, 0% 100%, ${clipX} 100%, ${clipX} ${clipY}, calc(${clipX} + ${clipWidth}) ${clipY}, calc(${clipX} + ${clipWidth}) calc(${clipY} + ${clipHeight}), ${clipX} calc(${clipY} + ${clipHeight}), ${clipX} 100%, 100% 100%, 100% 0%)`;
    return path;
  }
  return "";
}
__name(getClipPath, "getClipPath");

class DOMWidgetImpl {
  static {
    __name(this, "DOMWidgetImpl");
  }
  type;
  name;
  label;
  element;
  options;
  computedHeight;
  callback;
  height;
  cachedHeight;

  mouseDownHandler;
  constructor(name2, type, element, options4 = {}, height = 200) {
    this.type = type;
    this.name = name2;
    this.label = name2;
    this.element = element;
    this.options = options4;
    this.height = height;
    this.cachedHeight = height;

    if (element.blur) {
      this.mouseDownHandler = (event) => {
        if (!element.contains(event.target)) {
          element.blur();
        }
      };
      document.addEventListener("mousedown", this.mouseDownHandler);
    }
  }
  get value() {
    return this.options.getValue?.() ?? "";
  }
  set value(v2) {
    this.options.setValue?.(v2);
    this.callback?.(this.value);
  }
  /** Extract DOM widget size info */
  computeLayoutSize(node3) {
    const styles = getComputedStyle(this.element);
    let minHeight = this.options.getMinHeight?.() ?? parseInt(styles.getPropertyValue("--comfy-widget-min-height"));
    let maxHeight = this.options.getMaxHeight?.() ?? parseInt(styles.getPropertyValue("--comfy-widget-max-height"));
    let prefHeight = this.options.getHeight?.() ?? styles.getPropertyValue("--comfy-widget-height");
    if (typeof prefHeight === "string" && prefHeight.endsWith?.("%")) {
      prefHeight = node3.size[1] * (parseFloat(prefHeight.substring(0, prefHeight.length - 1)) / 100);
    } else {
      prefHeight = typeof prefHeight === "number" ? prefHeight : parseInt(prefHeight);
      if (isNaN(minHeight)) {
        minHeight = prefHeight;
      }
    }
    return {
      minHeight: isNaN(minHeight) ? 50 : minHeight,
      maxHeight: isNaN(maxHeight) ? void 0 : maxHeight,
      minWidth: 0
    };
  }
  draw(ctx, node3, widgetWidth, y2) {
    this.height = this.hideElement ? 0 : this.cachedHeight;
    this.computedHeight = this.hideElement ? 0 : this.cachedHeight;

    const { offset, scale } = app.canvas.ds;
    const hidden = !!this.options.hideOnZoom && app.canvas.low_quality || (this.computedHeight ?? 0) <= 0 || // @ts-expect-error custom widget type
    this.type === "converted-widget" || // @ts-expect-error custom widget type
    this.type === "hidden";


    this.element.dataset.shouldHide = hidden ? "true" : "false";
    const isInVisibleNodes = this.element.dataset.isInVisibleNodes === "true";
    const isCollapsed = this.element.dataset.collapsed === "true";

    const actualHidden = hidden || !isInVisibleNodes || isCollapsed; /* WE NEED TO UNDERSTAND WHY BY DEFAULT ACTUAL HIDDEN IS TRUE */
    const wasHidden = this.element.hidden;
    this.element.hidden = actualHidden;
    this.element.style.display = actualHidden ? "none" : "";

    if (actualHidden && !wasHidden) {
      this.options.onHide?.(this);
    }
    if (actualHidden) {
      return;
    }
    const elRect = ctx.canvas.getBoundingClientRect();
    const margin = 10;
    const top = node3.pos[0] + offset[0] + margin;
    const left = node3.pos[1] + offset[1] + margin + y2;
    Object.assign(this.element.style, {
      transformOrigin: "0 0",
      transform: `scale(${scale})`,
      left: `${top * scale}px`,
      top: `${left * scale}px`,
      width: `${widgetWidth - margin * 2}px`,
      height: `${(this.computedHeight ?? 50) - margin * 2}px`,
      position: "absolute",
      zIndex: app.graph.nodes.indexOf(node3),
      pointerEvents: app.canvas.read_only ? "none" : "auto"
    });
    const DOMClippingEnabled = true;
    if (DOMClippingEnabled) {
      const clipPath = getClipPath(node3, this.element, elRect);
      this.element.style.clipPath = clipPath ?? "none";
      this.element.style.willChange = "clip-path";
    }
    this.options.onDraw?.(this);
  }
  computeSize(){
    return [0, this.height];
  }
  onRemove() {
    if (this.mouseDownHandler) {
      document.removeEventListener("mousedown", this.mouseDownHandler);
    }
    this.element.remove();
  }
}

function distributeSpace(totalSpace, requests){
if (requests.length === 0) return [];
    const totalMinSize = requests.reduce((sum, req) => sum + req.minSize, 0);
    if (totalSpace < totalMinSize) {
        return requests.map((req) => req.minSize);
    }
    let allocations = requests.map((req) => ({
        computedSize: req.minSize,
        maxSize: req.maxSize ?? Infinity,
        remaining: (req.maxSize ?? Infinity) - req.minSize
    }));
    let remainingSpace = totalSpace - totalMinSize;
    while (remainingSpace > 0 && allocations.some((alloc) => alloc.remaining > 0)) {
        const growableItems = allocations.filter(
        (alloc) => alloc.remaining > 0
        ).length;
        if (growableItems === 0) break;
        const sharePerItem = remainingSpace / growableItems;
        let spaceUsedThisRound = 0;
        allocations = allocations.map((alloc) => {
        if (alloc.remaining <= 0) return alloc;
        const growth = Math.min(sharePerItem, alloc.remaining);
        spaceUsedThisRound += growth;
        return {
            ...alloc,
            computedSize: alloc.computedSize + growth,
            remaining: alloc.remaining - growth
        };
        });
        remainingSpace -= spaceUsedThisRound;
        if (spaceUsedThisRound === 0) break;
    }
    return allocations.map(({ computedSize }) => computedSize);
}

__name(distributeSpace, "distributeSpace");

//it's a copy of the computeSize function, because it doesn't account for the inputs that are hidden, se we basically do it with the right height after the resize
function computeSize(node){

    if (!node.widgets?.[0]?.last_y) return;
    let y2 = node.widgets[0].last_y;
    let freeSpace = node.size[1] - y2;
    let fixedWidgetHeight = 0;
    const layoutWidgets = [];
    for (const w2 of node.widgets) {
      if(w2.hidden || w2.hideElement) continue;
        if (w2.type === "converted-widget") {
        delete w2.computedHeight;
        } else if (w2.computeLayoutSize) {
        const { minHeight, maxHeight } = w2.computeLayoutSize(node);
        layoutWidgets.push({
            minHeight,
            prefHeight: maxHeight,
            w: w2
        });
        } else if (w2.computeSize) {
        fixedWidgetHeight += w2.computeSize()[1] + 4;
        } else {
        fixedWidgetHeight += LiteGraph.NODE_WIDGET_HEIGHT + 4;
        }
    }
    if (node.imgs && !node.widgets?.find((w2) => w2.name === ANIM_PREVIEW_WIDGET)) {
        fixedWidgetHeight += 220;
    }
    freeSpace -= fixedWidgetHeight;
    node.freeWidgetSpace = freeSpace;
    const spaceRequests = layoutWidgets.map((d2) => ({
        minSize: d2.minHeight,
        maxSize: d2.prefHeight
    }));
    const allocations = distributeSpace(Math.max(0, freeSpace), spaceRequests);
    layoutWidgets.forEach((d2, i2) => {
        d2.w.computedHeight = allocations[i2];
    });
    const totalNeeded = fixedWidgetHeight + allocations.reduce((sum, h2) => sum + h2, 0);
    if (totalNeeded > node.size[1] - node.widgets[0].last_y) {
        node.size[1] = totalNeeded + node.widgets[0].last_y;
        node.graph?.setDirtyCanvas(true);
    }
    for (const w2 of node.widgets) {
        if(w2.hidden || w2.hideElement) continue;
        w2.y = y2;
        if (w2.computedHeight) {
        y2 += w2.computedHeight;
        } else if (w2.computeSize) {
        y2 += w2.computeSize()[1] + 4;
        } else {
        y2 += LiteGraph.NODE_WIDGET_HEIGHT + 4;
        }
    }
}

__name(computeSize, "computeSize");

const addDOMWidget = (node, name, type, element, options = {}, height = 200) => {

    if(!elementWidgets)
      app.canvas.computeVisibleNodes([], []);

    options = { hideOnZoom: true, selectOn: ["focus", "click"], ...options };
    if (!element.parentElement) {
      app.canvasContainer.append(element);
    }
    element.hidden = true;
    element.style.display = "none";
    const { nodeData } = node.constructor;
    const tooltip = (nodeData?.input.required?.[name] ?? nodeData?.input.optional?.[name])?.[1]?.tooltip;
    if (tooltip && !element.title) {
      element.title = tooltip;
    }
    const widget = new DOMWidgetImpl(name, type, element, options, height);
    Object.defineProperty(widget, "value", {
      get() {
        return this.options.getValue?.() ?? "";
      },
      set(v2) {
        this.options.setValue?.(v2);
        this.callback?.(node.value);
      }
    });
    const selectEvents = options.selectOn ?? ["focus", "click"];
    for (const evt of selectEvents) {
      element.addEventListener(evt, () => {
        app.canvas.selectNode(node);
        app.canvas.bringToFront(node);
      });
    }
    node.addCustomWidget(widget);
    elementWidgets.add(node);
    const collapse = node.collapse;
    node.collapse = function(force) {
      collapse.call(node, force);
      if (node.collapsed) {
        element.hidden = true;
        element.style.display = "none";
      }
      element.dataset.collapsed = node.collapsed ? "true" : "false";
    };
    const { onConfigure } = node;
    node.onConfigure = function(serializedNode) {
      onConfigure?.call(node, serializedNode);
      element.dataset.collapsed = node.collapsed ? "true" : "false";
    };

    const onRemoved = node.onRemoved;
    node.onRemoved = function() {
      element.remove();
      elementWidgets.delete(node);
      onRemoved?.call(node);
    };
    if (!node[SIZE]) {
      node[SIZE] = true;
      const onResize2 = node.onResize;
      node.onResize = function(size) {
        options.beforeResize?.call(widget, node);
        computeSize(node, node.size)
        onResize2?.call(node, size);
        options.afterResize?.call(widget, node);
      };
    }

    node.widgets.pop();
    return widget;
  };

  export const computeNodeSize = (node, out) => {
    const ctorSize = node.constructor.size;
    if (ctorSize) return [ctorSize[0], ctorSize[1]];
    let rows = Math.max(
      node.inputs ? node.inputs.length : 1,
      node.outputs ? node.outputs.length : 1
    );
    const size = out || new Float32Array([0, 0]);
    rows = Math.max(rows, 1);
    const font_size = LiteGraph.NODE_TEXT_SIZE;
    const title_width = compute_text_size(node.title);
    let input_width = 0;
    let output_width = 0;
    if (node.inputs) {
      for (let i2 = 0, l2 = node.inputs.length; i2 < l2; ++i2) {
        const input = node.inputs[i2];
        const text2 = input.label || input.localized_name || input.name || "";
        const text_width = compute_text_size(text2);
        if (input_width < text_width)
          input_width = text_width;
      }
    }
    if (node.outputs) {
      for (let i2 = 0, l2 = node.outputs.length; i2 < l2; ++i2) {
        const output = node.outputs[i2];
        const text2 = output.label || output.localized_name || output.name || "";
        const text_width = compute_text_size(text2);
        if (output_width < text_width)
          output_width = text_width;
      }
    }
    size[0] = Math.max(input_width + output_width + 10, title_width);
    size[0] = Math.max(size[0], LiteGraph.NODE_WIDTH);
    if (node.widgets?.length)
      size[0] = Math.max(size[0], LiteGraph.NODE_WIDTH * 1.5);
    size[1] = (node.constructor.slot_start_y || 0) + rows * LiteGraph.NODE_SLOT_HEIGHT;
    let widgets_height = 0;
    if (node.widgets?.length) {
      for (let i2 = 0, l2 = node.widgets.length; i2 < l2; ++i2) {
        const widget = node.widgets[i2];
        if (widget.hideElement || widget.hidden || widget.advanced && !node.showAdvanced) continue;
        widgets_height += widget.computeSize ? widget.computeSize(size[0])[1] + 4 : LiteGraph.NODE_WIDGET_HEIGHT + 4;
      }
      widgets_height += 8;
    }
    if (node.widgets_up)
      size[1] = Math.max(size[1], widgets_height);
    else if (node.widgets_start_y != null)
      size[1] = Math.max(size[1], widgets_height + node.widgets_start_y);
    else
      size[1] += widgets_height;
    function compute_text_size(text2) {
      return text2 ? font_size * text2.length * 0.6 : 0;
    }
    __name(compute_text_size, "compute_text_size");
    if (node.constructor.min_height && size[1] < node.constructor.min_height) {
      size[1] = node.constructor.min_height;
    }
    size[1] += 6;
    return size;
  }

 export const addCustomTextWidget = (node, name, options = null, height = 200, interactible = true) =>{
    /*
    I hate myself for doing this but since the widgets classes aren't exposed, 
    and creating a widget object isn't enough to ensure the right behaviour for the widget, 
    we have to create a widget of our chosing via the standard addWidget, clone its class, and then instantiate our widget from there
    */
    const inputEl = document.createElement("textarea");
    inputEl.className = "comfy-multiline-input";
    inputEl.value = options.defaultVal;
    inputEl.placeholder = options.placeholder || name;
    inputEl.disabled = !interactible;
    
    if (app.vueAppReady) {
      inputEl.spellcheck = false;
    }

    const widget = addDOMWidget(node, name, "customtext", inputEl, {
      getValue() {
        return inputEl.value;
      },
      setValue(v2) {
        inputEl.value = v2;
      },
      getMinHeight(){
        return height;
      },
      getMaxHeight(){
        return height;
      },
      getHeight(){
        return height;
      },
    }, height);
    widget.inputEl = inputEl;
    widget.computedHeight = height;

    inputEl.addEventListener("input", () => {
      widget.callback?.(widget.value);
    });
    inputEl.addEventListener("pointerdown", (event) => {
      if (event.button === 1) {
        app.canvas.processMouseDown(event);
      }
    });
    inputEl.addEventListener("pointermove", (event) => {
      if ((event.buttons & 4) === 4) {
        app.canvas.processMouseMove(event);
      }
    });
    inputEl.addEventListener("pointerup", (event) => {
      if (event.button === 1) {
        app.canvas.processMouseUp(event);
      }
    });

    return widget;
}

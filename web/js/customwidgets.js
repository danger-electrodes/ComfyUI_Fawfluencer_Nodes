import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const multiline_text_widget_fixed_url = new URL("multilinetextwidgetfixed.js", import.meta.url).href;
const { addCustomTextWidget, computeNodeSize } = await import(multiline_text_widget_fixed_url);

const repeatCharacter = (characterToRepeat, count) => {
    let str = ""

    for(let i = 0; i < count; i++)
        str += characterToRepeat;

    return str;
}

const getSeparationWidget = (widget, widgetName, separatorWidth = 32) => {

    let diff = Math.max(0, ((separatorWidth - widgetName.length)/2) - 1);
    let halfBar = repeatCharacter("-", diff);

    let text = `${halfBar}[ ${widgetName} ]${halfBar}`;

    return  {
        type: "HTML",
        name: `separator_${widget.name}`,
        draw(ctx, node, width, y) {
            if (node.flags.collapsed) return;
    
            // Cache previous context properties
            const prevFont = ctx.font;
            const prevFillStyle = ctx.fillStyle;
            const prevTextAlign = ctx.textAlign;
            const prevTextBaseline = ctx.textBaseline;
    
            // Modify context for text drawing
            ctx.font = "16px Arial";
            ctx.fillStyle = "white";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";

            const textX = width / 2;
            const textY = y + 30; // Center text within the widget height (20px)
    
            // Draw the text
            ctx.fillText(text, textX, textY);
    
            // Restore previous context properties
            ctx.font = prevFont;
            ctx.fillStyle = prevFillStyle;
            ctx.textAlign = prevTextAlign;
            ctx.textBaseline = prevTextBaseline;
        },
        computeSize() {
            return [200, 60]; // Keeps the separator at a fixed height of 20px
        }
    };
}

const getWidget = (node, type, name, label, value = null, callback = null, options = null, force_hidden = false) =>{
        /*
        I hate myself for doing this but since the widgets classes aren't exposed, 
        and creating a widget object isn't enough to ensure the right behaviour for the widget, 
        we have to create a widget of our chosing via the standard addWidget, clone its class, and then instantiate our widget from there
        */
        let tempWidget = node.addWidget(type, name, value, callback, options, force_hidden);
        let WidgetClass = Object.getPrototypeOf(tempWidget).constructor;
        node.widgets.pop();

        let widgetTemplate = {
        // @ts-expect-error Type check or just assert?
        type: type.toLowerCase(),
        name: name,
        label: label,
        value: value,
        callback,
        options: options || {},
        force_hidden: force_hidden || false
        };
        let widget = new WidgetClass(widgetTemplate);
        return widget;
}

const uploadFile = async(widget, file) =>{
    try{
        console.log(widget);
        const body = new FormData();
        body.append("image", file);
    
        const resp = await api.fetchApi("/upload/image", {
            method: "POST",
            body
          });
    
        if (resp.status === 200) 
        {
            const data26 = await resp.json();
            let path = data26.name;
            if (data26.subfolder) path = data26.subfolder + "/" + path;
            if (!widget.options) {
                widget.options = { values: [] };
            }
            if (!widget.options.values) {
                widget.options.values = [];
            }
            if (!widget.options.values.includes(path)) {
                widget.options.values.push(path);
            }
            widget.value = path;
            widget.callback(widget.value);
        } 
        else 
        {
            let error = resp.status + " - " + resp.statusText;
            console.error(error);
            alert(error);
        }
    }
    catch(e)
    {
        let error = String(e);
        console.error(error);
        alert(error);
    }
}

const selectFile = (widget, accept = "image/jpeg, image/png, image/webp") => {
    // If the input element doesn't exist, create it
    let fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept = accept;

    // Add an event listener to handle file selection
    fileInput.addEventListener("change", async (event) => {
        const file = event.target.files[0]; // Get the selected file

        // Clean up the input element after file selection
        fileInput.value = ''; // Reset the value

        if (file) await uploadFile(widget, file);

        fileInput.remove();

    });

    // Trigger the file input dialog
    fileInput.click();
}

const refreshFolder = async(node, widget_name, hidden_inputs, folder_name) => {
    let body = {
        folder_name: folder_name
    }
    const resp = await api.fetchApi("/refresh_folder_input", {
        method: "POST",
        body: JSON.stringify(body)
    });
    
    const datas = await resp.json()

    let folders = datas.datas.folders;

    let nodes = app.graph._nodes;

    for(let i = 0; i < nodes.length; i++){
        let node = nodes[i];
        let widget = node.widgets.find((w) => w.name == "target_folder" && folders.indexOf(folder_name+"/") >= 0)

        if(widget)
        {
            console.log(widget)
            const updated_folder_input = node.widgets.find((w) => widget_name === w.name);
            updated_folder_input.value =  folders.indexOf(updated_folder_input.value) >= 0 ? updated_folder_input.value : folders[0]
            updated_folder_input.options.values = folders;
        }

    }

  /*  let node = nodes[event.detail.node_id];
    if(node) {
        const w = node.widgets.find((w) => event.detail.widget_name === w.name);
        if(w) {
            w.value = event.detail.value;
        }
    }

    let pathToImageFolder = hidden_inputs['path_to_image_folder'][1]['default'];
    console.log(pathToImageFolder)

    const w = node.widgets.find((w) => widget_name === w.name);
    w.value = "ok"
    w.options.values = []
    console.log(w)*/
}

//URL for null image
const not_found_url = new URL("assets/no_image_found.ico", import.meta.url).href;

//function to get the image url from the input folder (the one we provide when uploading an image)
const getImageURL = async (api, app, file_name, image_folder = "input") => {
    if (!file_name) {
        return not_found_url.href; // Return the default "not found" image if no filename is provided
    }

    // Extract subfolder (if exists)
    const folder_separator = file_name.lastIndexOf("/");
    let subfolder = "";
    if (folder_separator > -1) {
        subfolder = file_name.substring(0, folder_separator);
        file_name = file_name.substring(folder_separator + 1);
    }

    // Validate file extension
    const validExtensions = /\.(jpeg|jpg|png|webp)$/i;
    if (!validExtensions.test(file_name)) {
        return not_found_url.href;
    }

    // Construct image URL
    const url = api.apiURL(`/view?filename=${encodeURIComponent(file_name)}&type=${image_folder}&subfolder=${subfolder}${app.getPreviewFormatParam()}${app.getRandParam()}`);

    // Check if the image exists
    try {
        const response = await fetch(url, { method: "HEAD" }); // Use HEAD to check if file exists without downloading it
        if (response.ok) {
            return url; // File exists, return the URL
        } else {
            return not_found_url.href; // File does not exist, return default image
        }
    } catch (error) {
        console.error("Error checking image existence:", error);
        return not_found_url.href; // Return default image on error
    }
};

const fitAspect = (imageWidget) => {

    imageWidget.img.onload = () => {
        imageWidget.resizeX = imageWidget.img.naturalWidth <= imageWidget.img.naturalHeight;
        if(imageWidget.resizeX)
            imageWidget.ratio = imageWidget.img.naturalWidth/imageWidget.img.naturalHeight;
        else
            imageWidget.ratio = imageWidget.img.naturalHeight/imageWidget.img.naturalWidth;
    };
}

const displayImage = async (comboWidget, imageWidget, value, originalCallback = null) => {
    
    const src = comboWidget != undefined && comboWidget != null && value.trim() ? await getImageURL(api, app, value) : not_found_url;
    imageWidget.img.src = src;

    fitAspect(imageWidget);

    if (originalCallback)
        originalCallback(value);
};

/*function to create our custom widgets (in this case image widget and a button widget), 
we create them separately and we don't use Node.adWidget, we simply insert them where we want, 
very useful if we want a another widget to have more fields, 
in our case, every widget in our custom node whose name ends with "_fawpload" is a set of widget 
(a combo widget, with the filename in it to be retrieved when processing the images, this is the original widget, you can find it in the python node in the inputs declaration
an image preview widget (being returned as a first argument in imageUploadWidget())
and a button widget, to browse the files from the computer)*/
const imageUploadWidgetSection = (node, widget, widgetLabel, additionalWidgets = []) =>  {

	let imageWidget = {
		type: "HTML",
		name: `image_display_${widget.name}`,
		initialize(){

			this.img = document.createElement('img');

			let src = not_found_url;
			this.img.src = src;

            fitAspect(this);

			setTimeout(() => displayImage(widget, this, widget.value), 100);

			const originalCallback = widget.callback;
			widget.callback = (value) => {
				displayImage(widget, this, value, originalCallback);
			};
		},
		draw(ctx, node, width, y){
			if(node.flags.collapsed)
				return;
            if(this.resizeX)
            {
                const imageWidth = 200 * this.ratio;
                const x = (width - imageWidth) / 2; // Center the image
                ctx.drawImage(this.img, x, y, imageWidth, 200);
            }
            else
            {
                const imageHeight = 200 * this.ratio;
                const x = (width - 200)/2; // Center the image
                const yOffset = (200 - imageHeight) / 2;
                ctx.drawImage(this.img, x, y + yOffset, 200, imageHeight);
            }
		},
		computeSize(){
			return [200, 200];
		}
	};

	let buttonWidget = getWidget(node, "button", `upload_button_${widget.name}`, `UPLOAD ${widgetLabel} IMAGE`);

    buttonWidget.initialize = function(callback){
        this.callback = callback;
    }

    let toggleWidget = getWidget(node, "toggle", `enable_${widget.name}`, `enable ${widgetLabel.toLowerCase()}`, true)
    toggleWidget.initialize = function(callback){
        this.callback = callback;
    }

    let additionals = []

    for(let i = 0; i < additionalWidgets.length; i++)
    {
        let additionalWidgetObject = additionalWidgets[i];
        let additionalWidget = getWidget(node, additionalWidgetObject.type, additionalWidgetObject.name, additionalWidgetObject.label, additionalWidgetObject.defaultValue, additionalWidgetObject.callback, additionalWidgetObject.options, additionalWidgetObject.force_hidden);
        additionals.push(additionalWidget);
    }

	return {imageWidget, buttonWidget, toggleWidget, additionals}
};

const folderUploadWidgetSection = (node, widget, widgetLabel, additionalWidgets = []) =>  {

	let buttonWidget = getWidget(node, "button", `upload_button_${widget.name}`, `REFRESH ${widgetLabel}`);

    buttonWidget.initialize = function(callback){
        this.callback = callback;
    }

    let toggleWidget = getWidget(node, "toggle", `enable_${widget.name}`, `enable ${widgetLabel.toLowerCase()}`, true)
    toggleWidget.initialize = function(callback){
        this.callback = callback;
    }

    let additionals = []

    for(let i = 0; i < additionalWidgets.length; i++)
    {
        let additionalWidgetObject = additionalWidgets[i];
        let additionalWidget = getWidget(node, additionalWidgetObject.type, additionalWidgetObject.name, additionalWidgetObject.label, additionalWidgetObject.defaultValue, additionalWidgetObject.callback, additionalWidgetObject.options, additionalWidgetObject.force_hidden);
        additionals.push(additionalWidget);
    }

	return {buttonWidget, toggleWidget, additionals}
};


const promptWidgetSection = (node, widget, widgetLabel, additionalWidgets = []) =>  {
    let prompt_widget = addCustomTextWidget(node, `${widget.name}_conditionning`, {defaultVal: "", placeholder:"Describe the picture here. Use dynamic variables like '{clothes}' in 'text area' mode. Edit via 'URL_NOT_IMPLEMENTED_YET'.", dynamicPrompts: true, multiline: true}, 150);
    
    prompt_widget.initialize = function(callback){
        this.callback = callback;
    }

    let toggleWidget = getWidget(node, "toggle", `enable_${widget.name}`, `enable ${widgetLabel.toLowerCase()}`, true)
    toggleWidget.initialize = function(callback){
        this.callback = callback;
    }

    let additionals = []

    for(let i = 0; i < additionalWidgets.length; i++)
    {
        let additionalWidgetObject = additionalWidgets[i];
        let additionalWidget = getWidget(node, additionalWidgetObject.type, additionalWidgetObject.name, additionalWidgetObject.label, additionalWidgetObject.defaultValue, additionalWidgetObject.callback, additionalWidgetObject.options, additionalWidgetObject.force_hidden);
        additionals.push(additionalWidget);
    }


    let prompt_preview_widget = addCustomTextWidget(node, `${widget.name}_preview`, {defaultVal: "", placeholder:"Preview custom prompts and LLM text.", dynamicPrompts: true, multiline: true}, 100, false);

    let negative_prompt_widget = addCustomTextWidget(node, `negative_${widget.name}_conditionning`, {defaultVal: "", placeholder:"Negative here, what shouldn't be allowed in the image.", dynamicPrompts: true, multiline: true}, 150);
	return {prompt_widget, toggleWidget, additionals, prompt_preview_widget, negative_prompt_widget}
};

const insertWidgetAfter = (widgets, widget_to_add, index, callback = null) => {
    widgets.splice(index + 1, 0, widget_to_add);

    if (callback != null) callback();

    return index + 1; // Move to the next index
};

const insertWidgetBefore = (widgets, widget_to_add, index, callback = null) => {
    if (index === 0) {
        // If it's the first widget, insert at the beginning
        widgets.unshift(widget_to_add);
    } else {
        widgets.splice(index, 0, widget_to_add);
    }

    if (callback != null) callback();

    return index + 1; // Move to the next index
};

const resizeNodeDelay = (node, ms) =>{
    setTimeout(() => {resizeNode(node); console.log("ok")}, ms);
}

const resizeNode = (node) =>{
    console.log(node.computeSize)
    let size = computeNodeSize(node);
    node.setSize(size);
}


const hideElements = (node, hide, elements) =>{

    if(!elements)
        return;

    for (let i = 0; i < elements.length; i++)
    {
        let element = elements[i];

        if(element.force_hidden)
        {
            element.hidden = true;
            continue;
        }

        if(element.inputEl)
            element.hideElement = hide;
        else
            element.hidden = hide; 
    }
    resizeNode(node);
}

const hideElementsOnLoad = (node, toggleWidget, elements) => {
    setTimeout(() => {
        hideElements(node, !toggleWidget.value, elements);
    }, 100);
}

const isWidget = (_inputData, inputName, app) =>{
    const type = _inputData[0];
    const options = _inputData[1] ?? {};
    const inputData = [type, options];

    const widgetType = app.getWidgetType(inputData, inputName);

    return widgetType;
}

//returns an array of custom_datas, the items are undefined if no custom_datas specified
const getWidgetAdditionalDatas = (inputs) => {
    let additionalDatas = [];
    for(const inputName in inputs)
    {
        const inputData = inputs[inputName];
        const _isWidget = isWidget(inputData, inputName, app);

        if(!_isWidget)
            continue;

        additionalDatas.push(inputData[1]?.custom_datas);
    }

    return additionalDatas;
}

const addAdditionalCallbacksFawfluxencerNode = (node) => {
    const num_steps = node.widgets.find(w => w.name === "steps");
    const control_net_start_at = node.widgets.find(w => w.name === "control_net_start_at");
    const control_net_end_at = node.widgets.find(w => w.name === "control_net_end_at");
    const face_swap_step_start = node.widgets.find(w => w.name === "face_swap_step_start");
    
    setTimeout(() => { face_swap_step_start.options.max = num_steps.value;}, 100);
    num_steps.callback = (value) => 
    { 
        face_swap_step_start.options.max = value; 
        face_swap_step_start.value = Math.min(face_swap_step_start.value, value);
    }

    control_net_start_at.callback = (value) => {
        control_net_end_at.value = Math.max(value, control_net_end_at.value);
    }

    control_net_end_at.callback = (value) => {
        control_net_start_at.value = Math.min(value, control_net_start_at.value);
    }
}

const addAdditionalCallbacksImg2ImgFawfluencerNodeSDXL = (node) => {
    const control_net_start_at = node.widgets.find(w => w.name === "control_net_start_at");
    const control_net_end_at = node.widgets.find(w => w.name === "control_net_end_at");
    
    control_net_start_at.callback = (value) => {
        control_net_end_at.value = Math.max(value, control_net_end_at.value);
    }

    control_net_end_at.callback = (value) => {
        control_net_start_at.value = Math.min(value, control_net_start_at.value);
    }
}

const addImageUploadWidget = (node, widget, widget_custom_data, i) => 
{
    let widgetLabel = widget.label.replace("_", " ").toUpperCase();
    let separatorWidget = getSeparationWidget(widget, widgetLabel);
    let {imageWidget, buttonWidget, toggleWidget, additionals} = imageUploadWidgetSection(node, widget, widgetLabel, widget_custom_data.additional_widgets ?? []);
                            
    let elementsToHide = [widget, imageWidget, buttonWidget]
    elementsToHide = elementsToHide.concat(additionals)

    i = insertWidgetBefore(node.widgets, separatorWidget, i);

    if(widget_custom_data.is_optional)
        i = insertWidgetBefore(node.widgets, toggleWidget, i, () => toggleWidget.initialize((value) => hideElements(node, !value, elementsToHide)));

    i = insertWidgetAfter(node.widgets, imageWidget, i, () => {imageWidget.initialize();});
    i = insertWidgetAfter(node.widgets, buttonWidget, i, () => {buttonWidget.initialize(() => { selectFile(widget)})});

    for(let a = 0; a < additionals.length; a++)
        i = insertWidgetAfter(node.widgets, additionals[a], i);
    
    if(widget_custom_data.is_optional)
        hideElementsOnLoad(node, toggleWidget, elementsToHide);

    return i;
}

const addFolderUploadWidget = (node, widget, widget_custom_data, hidden_inputs, i) => 
{
    let widgetLabel = widget.label.replace("_", " ").toUpperCase();
    let separatorWidget = getSeparationWidget(widget, widgetLabel);
    let {buttonWidget, toggleWidget, additionals} = folderUploadWidgetSection(node, widget, widgetLabel, widget_custom_data.additional_widgets ?? []);
                            
    let elementsToHide = [widget, buttonWidget]
    elementsToHide = elementsToHide.concat(additionals)

    i = insertWidgetBefore(node.widgets, separatorWidget, i);

    if(widget_custom_data.is_optional)
        i = insertWidgetBefore(node.widgets, toggleWidget, i, () => toggleWidget.initialize((value) => hideElements(node, !value, elementsToHide)));

    i = insertWidgetAfter(node.widgets, buttonWidget, i, () => {buttonWidget.initialize(() => { refreshFolder(node, widget.name, hidden_inputs, "img_to_img_folder")})});

    for(let a = 0; a < additionals.length; a++)
        i = insertWidgetAfter(node.widgets, additionals[a], i);
    
    if(widget_custom_data.is_optional)
        hideElementsOnLoad(node, toggleWidget, elementsToHide);
    return i;
}

const addPromptWidget = (node, widget, widget_custom_data, i) => 
{
    let widgetLabel = widget.label.replace("_", " ").toUpperCase();
    let separatorWidget = getSeparationWidget(widget, widgetLabel);
    let {prompt_widget, toggleWidget, additionals, prompt_preview_widget, negative_prompt_widget} = promptWidgetSection(node, widget, widgetLabel, widget_custom_data.additional_widgets ?? []);
                   
    let elementsToHide = [widget, prompt_widget, prompt_preview_widget, negative_prompt_widget]
    elementsToHide = elementsToHide.concat(additionals)

    i = insertWidgetBefore(node.widgets, separatorWidget, i);

    if(widget_custom_data.is_optional)
        i = insertWidgetBefore(node.widgets, toggleWidget, i, () => toggleWidget.initialize((value) => hideElements(node, !value, elementsToHide)));

    i = insertWidgetAfter(node.widgets, prompt_widget, i);

    for(let a = 0; a < additionals.length; a++)
        i = insertWidgetAfter(node.widgets, additionals[a], i);

    if(widget_custom_data.allow_preview)
        i = insertWidgetAfter(node.widgets, prompt_preview_widget, i);

    if(widget_custom_data.is_optional)
        hideElementsOnLoad(node, toggleWidget, elementsToHide);

    if(widget_custom_data.allow_negative)
        i = insertWidgetAfter(node.widgets, negative_prompt_widget, i);

    const widgetOriginalCallback = widget.callback;

    widget.callback = (value) => {
        prompt_widget.value = "";

        switch(value){
            case "text area":
                prompt_widget.inputEl.placeholder = "Describe the picture here. Use dynamic variables like '{clothes}' i.e 'A beautiful woman wearing a {clothes}, the background is a {background}'. Edit the variables via 'URL_NOT_IMPLEMENTED_YET'.";
                break;

            case "llm generated text":
                prompt_widget.inputEl.placeholder = "Give your instructions to the llm here.";
                break;
        }
        if(widgetOriginalCallback)
            widgetOriginalCallback(value);
    }
    return i;
}

function nodeUpdatePromptHandler(event) {
	let nodes = app.graph._nodes_by_id;
	let node = nodes[event.detail.node_id];
	if(node) {
		const w = node.widgets.find((w) => event.detail.widget_name === w.name);
		if(w) {
			w.value = event.detail.value;
		}
	}
}

const addNodeCustomInputs = (nodeType, nodeData, callback = null) => {

    let inputs = nodeData["input"]["required"];
    let hidden_inputs = nodeData["input"]["hidden"];

    const additionalDatas = getWidgetAdditionalDatas(inputs)

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        originalOnNodeCreated?.apply(this, arguments);

        //necessary to link the custom datas to the original widgets

        console.log(this)
        console.log(this.widgets)
        let initial_widgets_index = 0;
        for (let i = 0; i < this.widgets.length; i++) {

            let widget_custom_data = additionalDatas[initial_widgets_index];
            let widget = this.widgets[i];

            initial_widgets_index++;
            if(!widget_custom_data)
                continue;

            switch(widget_custom_data.widget_template){
                case "file_upload":
                    i = addImageUploadWidget(this, widget, widget_custom_data, i);
                    break;
                case "folder_upload":
                    i = addFolderUploadWidget(this, widget, widget_custom_data, hidden_inputs, i);
                    break;
                case "prompt":
                    i = addPromptWidget(this, widget, widget_custom_data, i);
                    break;
            }
        }

        
        let button = this.addWidget("button", "DEBUG", null, () => {
            console.log(this)
            console.log("FOr Debug purpose");
            console.log(this.widgets)
            console.log(this.inputs)
        });

        if(callback)
            callback(this);

        let hidden_widgets = [];

        for(let i  = 0; i < this.widgets.length; i++)
        {
            let widget = this.widgets[i];

            if(widget.force_hidden)
                hidden_widgets.push(widget);
        }

        hideElements(this, false, hidden_widgets);
    }
}

const registerNode = (extension) => {
    app.registerExtension({
        name: "Fawfulized."+extension.extensionName,

        //registering fawfulized-feedback so we can set the values of the widgets from the server
        async setup() {
            if(extension.setupCallback)
                extension.setupCallback();
        },

        //adding custom widgets/inputs
        async beforeRegisterNodeDef(nodeType, nodeData, app) {

            if (!nodeData?.category?.startsWith("Fawfulized") || nodeData.name !== extension.extensionName) {
                return;
            }
            console.log("ok..")
            addNodeCustomInputs(nodeType, nodeData, extension.additionalCallback);
        },
    });
}

export const checkAndAddCustomWidgets = () => {

    const FawfluxencerNode = {
        extensionName : "FawfluxencerNode",
        setupCallback : () => app.api.addEventListener("fawfulized-prompt-update", nodeUpdatePromptHandler),
        additionalCallback: (node) => addAdditionalCallbacksFawfluxencerNode(node)
    }

    const Img2ImgFawfluencerNodeSDXL = {
        extensionName : "Img2ImgFawfluencerNodeSDXL",
        additionalCallback: (node) => addAdditionalCallbacksImg2ImgFawfluencerNodeSDXL(node)
    }
    registerNode(FawfluxencerNode);
    registerNode(Img2ImgFawfluencerNodeSDXL);
}

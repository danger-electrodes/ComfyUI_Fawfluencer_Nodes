const custom_widget_url = new URL("customwidgets.js", import.meta.url).href;

const { checkAndAddCustomWidgets } = await import(custom_widget_url);
checkAndAddCustomWidgets();
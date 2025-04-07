from PIL import ExifTags
import piexif
import os
from .utilities import model_management as model_management_utilities

bboxs_model_path = model_management_utilities.load_model("ultralytics", "bbox")

BBOX_MODELS = [f for f in os.listdir(bboxs_model_path)]

LIGHTS = [
    {
        "name": "sunlight",
        "start_color_min": "#E69500",
        "start_color_max": "#FFB733",
        "end_color_min": "#000000",
        "end_color_max": "#000000"
    },
    {
        "name": "moonlight",
        "start_color_min": "#A0B8D0",
        "start_color_max": "#C0D8F0",
        "end_color_min": "#000000",
        "end_color_max": "#000000"
    },
    {
        "name": "red light",
        "start_color_min": "#CC0000",
        "start_color_max": "#FF3333",
        "end_color_min": "#000000",
        "end_color_max": "#000000"
    },
    {
        "name": "green light",
        "start_color_min": "#00CC00",
        "start_color_max": "#33FF33",
        "end_color_min": "#000000",
        "end_color_max": "#000000"
    },
    {
        "name": "blue light",
        "start_color_min": "#0000CC",
        "start_color_max": "#3333FF",
        "end_color_min": "#000000",
        "end_color_max": "#000000"
    },
    {
        "name": "yellow light",
        "start_color_min": "#CCCC00",
        "start_color_max": "#FFFF33",
        "end_color_min": "#000000",
        "end_color_max": "#000000"
    },
    {
        "name": "disco light",
        "start_color_min": "#CC1073",
        "start_color_max": "#FF2BAA",
        "end_color_min": "#000000",
        "end_color_max": "#000000"
    },
    {
        "name": "purple light",
        "start_color_min": "#660066",
        "start_color_max": "#993399",
        "end_color_min": "#000000",
        "end_color_max": "#000000"
    },
    {
        "name": "pink light",
        "start_color_min": "#E6A8B5",
        "start_color_max": "#FFC8D8",
        "end_color_min": "#000000",
        "end_color_max": "#000000"
    },
    {
        "name": "orange light",
        "start_color_min": "#E6925A",
        "start_color_max": "#FFB088",
        "end_color_min": "#000000",
        "end_color_max": "#000000"
    }
]

LIGHTS_INTENSITIES = [
    "ambient – a gentle presence of light that softly fills the space, casting minimal shadows",
    "faint – a weak and subtle glow, leaving deep shadows and a sense of dimness",
    "bright – noticeable but not overwhelming, illuminating surroundings while still allowing soft shadows",
    "soft – diffused and delicate, creating smooth transitions between light and shadow",
    "scattered – unevenly spread light, forming shifting shadows and areas of varying brightness",
    "harsh – strong directional light that casts sharp, defined shadows and stark contrasts",
    "glowing – a steady and contained luminance, with edges that blend subtly into darkness",
    "dull – a muted and subdued light, leaving behind lingering shadows and a dim atmosphere",
    "piercing – a focused and narrow beam, cutting through darkness and creating long, deep shadows",
    "warm – a comforting, golden hue that softens edges and gently fades into shadows",
    "cold – a crisp and bluish tone that enhances sharpness and makes shadows feel deeper",
    "diffused – evenly spread light with no harsh edges, softening shadows and reducing contrast",
    "pulsing – rhythmic fluctuations in brightness, causing shadows to shift subtly over time",
    "radiant – a steady, luminous presence that brightens without overpowering, leaving gentle shadows",
    "subtle – an understated glow that enhances visibility without erasing natural darkness"
]

CONTROL_NETS = [
                "openpose", 
                "depth", 
                "normal",
                "tile",
                "segment",
                "anyline",
                "metric3Dnormalmap",
                ]

ASPECT_RATIOS = [
            "256x256 (1:1)",
            "256x384 (2:3)",
            "384x256 (3:2)",
            "320x448 (5:7)",
            "448x320 (7:5)",
            "384x512 (3:4)",
            "512x384 (4:3)",
            "512x512 (1:1)",
            "512x768 (2:3)",
            "768x512 (3:2)",
            "512x640 (4:5)",
            "640x512 (5:4)",
            "448x768 (9:16)",
            "768x448 (16:9)",
            "576x1024 (9:16)",
            "704x1408 (1:2)", 
            "704x1344 (13:25)",
            "768x1344 (4:7)",
            "768x1280 (3:5)",
            "832x1216 (2:3)",
            "832x1152 (5:7)",
            "896x1152 (7:9)",
            "896x1088 (4:5)",
            "960x1088 (8:9)",
            "960x1024 (15:16)",
            "1024x1024 (1:1)",
            "1024x960 (16:15)",
            "1088x960 (9:8)",
            "1088x896 (5:4)",
            "1152x896 (9:7)",
            "1152x832 (7:5)",
            "1216x832 (3:2)",
            "1280x768 (5:3)",
            "1344x768 (7:4)",
            "1344x704 (16:9)",
            "1024x576 (16:9)",
            "1408x704 (2:1)",
            "1472x704 (21:10)",
            "1536x640 (12:5)",
            "1600x640 (5:2)",
            "1664x576 (3:1)",
            "1728x576 (3:1)"
        ]

#We are able to define custom datas for our inputs such as additional fields (for our custom image upload implementation for example, see how they are processed in "web/js/custom_widgets.js") like so : 
CONTROL_NET_CUSTOM_DATAS = {
    "widget_template" : "file_upload",
    "is_optional" : True,
    "additional_widgets" : [
                        {
                            "type" : "combo",
                            "name" : "control_net_type",
                            "label" : "control net type :",
                            "defaultValue": "tile",
                            "options": {
                                "values": CONTROL_NETS
                            }
                        },
                        {
                            "type" : "number",
                            "name" : "control_net_strength",
                            "label" : "control net strength :",
                            "defaultValue": 0.5,
                            "options": {
                                "min": 0.0,
                                "max": 1.0,
                                "precision": 2,
                                "step": 0.05,
                            }
                        },
                        {
                            "type" : "number",
                            "name" : "control_net_start_at",
                            "label" : "start at :",
                            "defaultValue": 0.0,
                            "options": {
                                "min": 0.0,
                                "max": 1.0,
                                "precision": 2,
                                "step": 0.05,
                            }
                        },
                        {
                            "type" : "number",
                            "name" : "control_net_end_at",
                            "label" : "end at :",
                            "defaultValue": 1.0,
                            "options": {
                                "min": 0.0,
                                "max": 1.0,
                                "precision": 2,
                                "step": 0.05,
                            }
                        },
                    ]
}

INFLUENCER_FACE_CUSTOM_DATAS = {
    "widget_template" : "file_upload",
    "is_optional" : False,
    "additional_widgets" : [
                        {
                            "type" : "number",
                            "name" : "face_swap_step_start",
                            "label" : "face swap starts at step :",
                            "defaultValue": 13,
                            "options": {
                                "min": 5,
                                "max": 15,
                                "precision": 0,
                                "step": 1,
                            }
                        },
                        {
                            "type" : "combo",
                            "name" : "insight_face_provider",
                            "label" : "provider for insightface :",
                            "defaultValue": "CPU",
                            "options": {
                                "values": ["CPU", "CUDA", "ROCM"]
                            }
                        },
                        {
                            "type" : "number",
                            "name" : "pulid_strength",
                            "label" : "pulid strength :",
                            "defaultValue": 0.90,
                            "options": {
                                "min": 0.00,
                                "max": 1.00,
                                "precision": 2,
                                "step": 0.01,
                            }
                        },
                        {
                            "type" : "number",
                            "name" : "insight_face_input_width",
                            "label" : "insightface input width :",
                            "defaultValue": 640,
                            "options": {
                                "min": 0,
                                "max": 1024,
                                "precision": 0,
                                "step": 1,
                            }
                        },
                        {
                            "type" : "number",
                            "name" : "insight_face_input_height",
                            "label" : "insightface input height :",
                            "defaultValue": 640,
                            "options": {
                                "min": 0,
                                "max": 1024,
                                "precision": 0,
                                "step": 1,
                            }
                        },
                    ]
}

INFLUENCER_BODY_CUSTOM_DATAS = {
    "widget_template" : "file_upload",
    "is_optional" : True,
    "additional_widgets" : []
}

BACKGROUND_CUSTOM_DATAS = {
    "widget_template" : "file_upload",
    "is_optional" : True,
    "additional_widgets" : [   
        {
            "type" : "combo",
            "name" : "mask_type",
            "label" : "mask type :",
            "defaultValue": "rectangle",
            "options": {
                "values": ["rectangle", "silhouette"]
            }
        },                     
        {
            "type" : "number",
            "name" : "background_denoise",
            "label" : "background denoise :",
            "defaultValue": 1.0,
            "options": {
                "min": 0.0,
                "max": 1.0,
                "precision": 2,
                "step": 0.01,
            }
        },
        {
            "type" : "number",
            "name" : "mask_grow_by",
            "label" : "grow mask by :",
            "defaultValue": 5,
            "options": {
                "min": 0,
                "max": 20,
                "precision": 0,
                "step": 1,
            }
        },
        {
            "type" : "number",
            "name" : "mask_blur_size",
            "label" : "mask blur size :",
            "defaultValue": 5.5,
            "options": {
                "min": 0,
                "max": 30,
                "precision": 2,
                "step": 0.5,
            }
        }
    ]
}

PROMPT_CUSTOM_DATAS = {
    "widget_template" : "prompt",
    "is_optional" : False,
    "allow_preview" : True,
    "allow_negative" : True,
    "additional_widgets" : []
}


########################################################
#####################[IMG 2 IMG]########################
########################################################

IMG2IMG_FOLDER_CUSTOM_DATAS = {
    "widget_template" : "folder_upload",
    "is_optional" : False,
    "additional_widgets" : [   
        {
            "type" : "number",
            "name" : "mask_grow_by",
            "label" : "grow mask by :",
            "defaultValue": 5,
            "options": {
                "min": 0,
                "max": 20,
                "precision": 0,
                "step": 1,
            }
        },
        {
            "type" : "number",
            "name" : "mask_blur_size",
            "label" : "mask blur size :",
            "defaultValue": 5.5,
            "options": {
                "min": 0,
                "max": 30,
                "precision": 2,
                "step": 0.5,
            }
        },
        {
            "type" : "string",
            "name" : "target_folder",
            "label" : "target folder :",
            "force_hidden" : True,
            "defaultValue" : "img_to_img_folder"
        }
    ]
}

IMG2IMG_INFLUENCER_FACE_CUSTOM_DATAS = {
    "widget_template" : "file_upload",
    "is_optional" : False,
    "additional_widgets" : [
                        {
                            "type" : "combo",
                            "name" : "insight_face_provider",
                            "label" : "provider for insightface :",
                            "defaultValue": "CPU",
                            "options": {
                                "values": ["CPU", "CUDA", "ROCM"]
                            }
                        },
                        {
                            "type" : "number",
                            "name" : "instantid_strength",
                            "label" : "instantid strength :",
                            "defaultValue": 0.90,
                            "options": {
                                "min": 0.00,
                                "max": 1.00,
                                "precision": 2,
                                "step": 0.01,
                            }
                        },
                        {
                            "type" : "number",
                            "name" : "instantid_cnet_strength",
                            "label" : "instantid controlnet strength :",
                            "defaultValue": 0.30,
                            "options": {
                                "min": 0.00,
                                "max": 1.00,
                                "precision": 2,
                                "step": 0.01,
                            }
                        },
                        {
                            "type" : "number",
                            "name" : "ip_strength",
                            "label" : "ip adapter strength :",
                            "defaultValue": 0.30,
                            "options": {
                                "min": 0.00,
                                "max": 1.00,
                                "precision": 2,
                                "step": 0.01,
                            }
                        },
                        {
                            "type" : "combo",
                            "name" : "instantid_combine_embeddings",
                            "label" : "instantid combine embeddings :",
                            "defaultValue": "average",
                            "options": {
                                "values": ["average", "norm average", "concat"]
                            }
                        },
                        {
                            "type" : "number",
                            "name" : "insight_face_input_width",
                            "label" : "insightface input width :",
                            "defaultValue": 640,
                            "options": {
                                "min": 0,
                                "max": 1024,
                                "precision": 0,
                                "step": 1,
                            }
                        },
                        {
                            "type" : "number",
                            "name" : "insight_face_input_height",
                            "label" : "insightface input height :",
                            "defaultValue": 640,
                            "options": {
                                "min": 0,
                                "max": 1024,
                                "precision": 0,
                                "step": 1,
                            }
                        },
                    ]
}
"""Maps EXIF tags to tag names."""
REV_TAGS = {value: key for key, value in ExifTags.TAGS.items()}

#"2025:01:29 20:22:58" DATE TIME
#"705871" MS
PHONES = [
    {
        "name": "HONOR 90 Lite",
        "metadatas" :        
         {
            "0th": {
                piexif.ImageIFD.DocumentName: "",
                piexif.ImageIFD.YCbCrPositioning: "1",
                piexif.ImageIFD.XResolution: "72/1",
                piexif.ImageIFD.YResolution: "72/1",
                piexif.ImageIFD.ResolutionUnit: "2",
                piexif.ImageIFD.ImageWidth: "__WIDTH__",
                piexif.ImageIFD.ImageLength: "__HEIGHT__",
                piexif.ImageIFD.Make: "HONOR",
                piexif.ImageIFD.Model: "CRT-NX1",
                piexif.ImageIFD.Orientation: "1",
                piexif.ImageIFD.Software: "CRT-N31 8.0.0.217(C431E9R2P2)",
                piexif.ImageIFD.DateTime: "__DATE__ __TIME__",
            },
            "Exif": {
                piexif.ExifIFD.ExposureTime: "1/14",
                piexif.ExifIFD.FNumber: "245/100",
                piexif.ExifIFD.ExposureProgram: "2",
                piexif.ExifIFD.ISOSpeedRatings: "6657",
                piexif.ExifIFD.ExifVersion: b"0210",
                piexif.ExifIFD.DateTimeOriginal: "__DATE__ __TIME__",
                piexif.ExifIFD.DateTimeDigitized: "__DATE__ __TIME__",
                piexif.ExifIFD.ShutterSpeedValue: "1/536870912",
                piexif.ExifIFD.ApertureValue: "245/100",
                piexif.ExifIFD.BrightnessValue: "-2/1",
                piexif.ExifIFD.ExposureBiasValue: "0/1",
                piexif.ExifIFD.MeteringMode: "2",
                piexif.ExifIFD.LightSource: "1",
                piexif.ExifIFD.Flash: "95",
                piexif.ExifIFD.FocalLength: "0/1",
                piexif.ExifIFD.SubSecTime: "__MS__",
                piexif.ExifIFD.SubSecTimeOriginal: "__MS__",
                piexif.ExifIFD.SubSecTimeDigitized: "__MS__",
                piexif.ExifIFD.FlashpixVersion: b"0100",
                piexif.ExifIFD.ColorSpace: "1",
                piexif.ExifIFD.PixelXDimension: "__WIDTH__",  # Correct field for width in Exif
                piexif.ExifIFD.PixelYDimension: "__HEIGHT__",  # Correct field for height in Exif
                piexif.ExifIFD.SensingMethod: "2",
                piexif.ExifIFD.FileSource: b"3",
                piexif.ExifIFD.SceneType: b"1",
                piexif.ExifIFD.CustomRendered: "1",
                piexif.ExifIFD.ExposureMode: "0",
                piexif.ExifIFD.WhiteBalance: "0",
                piexif.ExifIFD.FocalLengthIn35mmFilm: "24",
                piexif.ExifIFD.SceneCaptureType: "0",
                piexif.ExifIFD.GainControl: "0",
                piexif.ExifIFD.Contrast: "0",
                piexif.ExifIFD.Saturation: "0",
                piexif.ExifIFD.Sharpness: "0",
                piexif.ExifIFD.SubjectDistanceRange: "0",
            },
            "GPS": {},  # No GPS data provided
            #"1st": {},  # No thumbnail data provided
            "Interop": {
                piexif.InteropIFD.InteroperabilityIndex: "R98"
            },
            "thumbnail": {}
        }
    },
    {
        "name" : "IPHONE 6",
        "metadatas" :
        {
            "0th": {
                piexif.ImageIFD.DocumentName: "",
                piexif.ImageIFD.YCbCrPositioning: "1",
                piexif.ImageIFD.XResolution: "72/1",
                piexif.ImageIFD.YResolution: "72/1",
                piexif.ImageIFD.ResolutionUnit: "2",
                piexif.ImageIFD.ImageWidth: "__WIDTH__",
                piexif.ImageIFD.ImageLength: "__HEIGHT__",
                piexif.ImageIFD.Make: "HONOR",
                piexif.ImageIFD.Model: "CRT-NX1",
                piexif.ImageIFD.Orientation: "1",
                piexif.ImageIFD.Software: "CRT-N31 8.0.0.217(C431E9R2P2)",
                piexif.ImageIFD.DateTime: "__DATE__ __TIME__",
            },
            "Exif": {
                piexif.ExifIFD.ExposureTime: "1/14",
                piexif.ExifIFD.FNumber: "245/100",
                piexif.ExifIFD.ExposureProgram: "2",
                piexif.ExifIFD.ISOSpeedRatings: "6657",
                piexif.ExifIFD.ExifVersion: b"0210",
                piexif.ExifIFD.DateTimeOriginal: "__DATE__ __TIME__",
                piexif.ExifIFD.DateTimeDigitized: "__DATE__ __TIME__",
                piexif.ExifIFD.ShutterSpeedValue: "1/536870912",
                piexif.ExifIFD.ApertureValue: "245/100",
                piexif.ExifIFD.BrightnessValue: "-2/1",
                piexif.ExifIFD.ExposureBiasValue: "0/1",
                piexif.ExifIFD.MeteringMode: "2",
                piexif.ExifIFD.LightSource: "1",
                piexif.ExifIFD.Flash: "95",
                piexif.ExifIFD.FocalLength: "0/1",
                piexif.ExifIFD.SubSecTime: "__MS__",
                piexif.ExifIFD.SubSecTimeOriginal: "__MS__",
                piexif.ExifIFD.SubSecTimeDigitized: "__MS__",
                piexif.ExifIFD.FlashpixVersion: b"0100",
                piexif.ExifIFD.ColorSpace: "1",
                piexif.ExifIFD.PixelXDimension: "__WIDTH__",  # Correct field for width in Exif
                piexif.ExifIFD.PixelYDimension: "__HEIGHT__",  # Correct field for height in Exif
                piexif.ExifIFD.SensingMethod: "2",
                piexif.ExifIFD.FileSource: b"3",
                piexif.ExifIFD.SceneType: b"1",
                piexif.ExifIFD.CustomRendered: "1",
                piexif.ExifIFD.ExposureMode: "0",
                piexif.ExifIFD.WhiteBalance: "0",
                piexif.ExifIFD.FocalLengthIn35mmFilm: "24",
                piexif.ExifIFD.SceneCaptureType: "0",
                piexif.ExifIFD.GainControl: "0",
                piexif.ExifIFD.Contrast: "0",
                piexif.ExifIFD.Saturation: "0",
                piexif.ExifIFD.Sharpness: "0",
                piexif.ExifIFD.SubjectDistanceRange: "0",
            },
            "GPS": {},  # No GPS data provided
            #"1st": {},  # No thumbnail data provided
            "Interop": {
                piexif.InteropIFD.InteroperabilityIndex: "R98"
            },
            "thumbnail": {}
        }
    }
]
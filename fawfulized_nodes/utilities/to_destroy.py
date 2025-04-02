def apply_pulid_flux(self, model, pulid_flux, eva_clip, face_analysis, image, weight, start_at, end_at, prior_image=None,fusion="mean", fusion_weight_max=1.0, fusion_weight_min=0.0, train_step=1000, use_gray=True, attn_mask=None, unique_id=None):
        device = comfy.model_management.get_torch_device()
        # Why should I care what args say, when the unet model has a different dtype?!
        # Am I missing something?!
        #dtype = comfy.model_management.unet_dtype()
        dtype = model.model.diffusion_model.dtype
        # For 8bit use bfloat16 (because ufunc_add_CUDA is not implemented)
        if dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            dtype = torch.bfloat16

        eva_clip.to(device, dtype=dtype)
        pulid_flux.to(device, dtype=dtype)

        # TODO: Add masking support!
        if attn_mask is not None:
            if attn_mask.dim() > 3:
                attn_mask = attn_mask.squeeze(-1)
            elif attn_mask.dim() < 3:
                attn_mask = attn_mask.unsqueeze(0)
            attn_mask = attn_mask.to(device, dtype=dtype)

        if prior_image is not None:
            prior_image = resize_with_pad(prior_image.to(image.device, dtype=image.dtype), target_size=(image.shape[1], image.shape[2]))
            image=torch.cat((prior_image,image),dim=0)
        image = tensor_to_image(image)

        face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            device=device,
        )

        face_helper.face_parse = None
        face_helper.face_parse = init_parsing_model(model_name='bisenet', device=device)

        bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
        cond = []

        # Analyse multiple images at multiple sizes and combine largest area embeddings
        for i in range(image.shape[0]):
            # get insightface embeddings
            iface_embeds = None
            for size in [(size, size) for size in range(640, 256, -64)]:
                face_analysis.det_model.input_size = size
                face_info = face_analysis.get(image[i])
                if face_info:
                    # Only use the maximum face
                    # Removed the reverse=True from original code because we need the largest area not the smallest one!
                    # Sorts the list in ascending order (smallest to largest),
                    # then selects the last element, which is the largest face
                    face_info = sorted(face_info, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))[-1]
                    iface_embeds = torch.from_numpy(face_info.embedding).unsqueeze(0).to(device, dtype=dtype)
                    break
            else:
                # No face detected, skip this image
                logging.warning(f'Warning: No face detected in image {str(i)}')
                continue

            # get eva_clip embeddings
            face_helper.clean_all()
            face_helper.read_image(image[i])
            face_helper.get_face_landmarks_5(only_center_face=True)
            face_helper.align_warp_face()

            if len(face_helper.cropped_faces) == 0:
                # No face detected, skip this image
                continue

            # Get aligned face image
            align_face = face_helper.cropped_faces[0]
            # Convert bgr face image to tensor
            align_face = image_to_tensor(align_face).unsqueeze(0).permute(0, 3, 1, 2).to(device)
            parsing_out = face_helper.face_parse(functional.normalize(align_face, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[0]
            parsing_out = parsing_out.argmax(dim=1, keepdim=True)
            bg = sum(parsing_out == i for i in bg_label).bool()
            white_image = torch.ones_like(align_face)
            # Only keep the face features
            if use_gray:
                _align_face = to_gray(align_face)
            else:
                _align_face = align_face
            face_features_image = torch.where(bg, white_image, _align_face)

            # Transform img before sending to eva_clip
            # Apparently MPS only supports NEAREST interpolation?
            face_features_image = functional.resize(face_features_image, eva_clip.image_size, transforms.InterpolationMode.BICUBIC if 'cuda' in device.type else transforms.InterpolationMode.NEAREST).to(device, dtype=dtype)
            face_features_image = functional.normalize(face_features_image, eva_clip.image_mean, eva_clip.image_std)

            # eva_clip
            id_cond_vit, id_vit_hidden = eva_clip(face_features_image, return_all_features=False, return_hidden=True, shuffle=False)
            id_cond_vit = id_cond_vit.to(device, dtype=dtype)
            for idx in range(len(id_vit_hidden)):
                id_vit_hidden[idx] = id_vit_hidden[idx].to(device, dtype=dtype)

            id_cond_vit = torch.div(id_cond_vit, torch.norm(id_cond_vit, 2, 1, True))

            # Combine embeddings
            id_cond = torch.cat([iface_embeds, id_cond_vit], dim=-1)

            # Pulid_encoder
            cond.append(pulid_flux.get_embeds(id_cond, id_vit_hidden))

        if not cond:
            # No faces detected, return the original model
            logging.warning("PuLID warning: No faces detected in any of the given images, returning unmodified model.")
            return (model,)

        # fusion embeddings
        if fusion == "mean":
            cond = torch.cat(cond).to(device, dtype=dtype) # N,32,2048
            if cond.shape[0] > 1:
                cond = torch.mean(cond, dim=0, keepdim=True)
        elif fusion == "concat":
            cond = torch.cat(cond, dim=1).to(device, dtype=dtype)
        elif fusion == "max":
            cond = torch.cat(cond).to(device, dtype=dtype)
            if cond.shape[0] > 1:
                cond = torch.max(cond, dim=0, keepdim=True)[0]
        elif fusion == "norm_id":
            cond = torch.cat(cond).to(device, dtype=dtype)
            if cond.shape[0] > 1:
                norm=torch.norm(cond,dim=(1,2))
                norm=norm/torch.sum(norm)
                cond=torch.einsum("wij,w->ij",cond,norm).unsqueeze(0)
        elif fusion == "max_token":
            cond = torch.cat(cond).to(device, dtype=dtype)
            if cond.shape[0] > 1:
                norm=torch.norm(cond,dim=2)
                _,idx=torch.max(norm,dim=0)
                cond=torch.stack([cond[j,i] for i,j in enumerate(idx)]).unsqueeze(0)
        elif fusion == "auto_weight": # 🤔
            cond = torch.cat(cond).to(device, dtype=dtype)
            if cond.shape[0] > 1:
                norm=torch.norm(cond,dim=2)
                order=torch.argsort(norm,descending=False,dim=0)
                regular_weight=torch.linspace(fusion_weight_min,fusion_weight_max,norm.shape[0]).to(device, dtype=dtype)

                _cond=[]
                for i in range(cond.shape[1]):
                    o=order[:,i]
                    _cond.append(torch.einsum('ij,i->j',cond[:,i,:],regular_weight[o]))
                cond=torch.stack(_cond,dim=0).unsqueeze(0)
        elif fusion == "train_weight":
            cond = torch.cat(cond).to(device, dtype=dtype)
            if cond.shape[0] > 1:
                if train_step > 0:
                    with torch.inference_mode(False):
                        cond = online_train(cond, device=cond.device, step=train_step)
                else:
                    cond = torch.mean(cond, dim=0, keepdim=True)

        sigma_start = model.get_model_object("model_sampling").percent_to_sigma(start_at)
        sigma_end = model.get_model_object("model_sampling").percent_to_sigma(end_at)

        # Patch the Flux model (original diffusion_model)
        # Nah, I don't care for the official ModelPatcher because it's undocumented!
        # I want the end result now, and I don’t mind if I break other custom nodes in the process. 😄
        flux_model = model.model.diffusion_model
        # Let's see if we already patched the underlying flux model, if not apply patch
        if not hasattr(flux_model, "pulid_ca"):
            # Add perceiver attention, variables and current node data (weight, embedding, sigma_start, sigma_end)
            # The pulid_data is stored in Dict by unique node index,
            # so we can chain multiple ApplyPulidFlux nodes!
            flux_model.pulid_ca = pulid_flux.pulid_ca
            flux_model.pulid_double_interval = pulid_flux.double_interval
            flux_model.pulid_single_interval = pulid_flux.single_interval
            flux_model.pulid_data = {}
            # Replace model forward_orig with our own
            new_method = forward_orig.__get__(flux_model, flux_model.__class__)
            setattr(flux_model, 'forward_orig', new_method)

        # Patch is already in place, add data (weight, embedding, sigma_start, sigma_end) under unique node index
        flux_model.pulid_data[unique_id] = {
            'weight': weight,
            'embedding': cond,
            'sigma_start': sigma_start,
            'sigma_end': sigma_end,
        }

        # Keep a reference for destructor (if node is deleted the data will be deleted as well)
        self.pulid_data_dict = {'data': flux_model.pulid_data, 'unique_id': unique_id}

        return (model,)
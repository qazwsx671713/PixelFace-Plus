## Please init the pretrain Model first, and the use the following code to generate image
## ...
##

## if you have init model, you can use the code below to generate your own image
## you can add this code to the \code\trainer.py to use
def sampling(self):
  ## set device
  device = self.args.device
  
  ## init noise
  ## if you just want to generate 2 image,
  ## then set self.args.n_sample = 2
  ## self.args.latent usually set 512
  ori_noise = [torch.randn(self.args.n_sample, self.args.latent, device=device)]
  
  ## set captions, the number of captions should equal to self.args.n_sample
  ## if you just want to generate 2 image, you should only use 2 captions
  caps = ['caption_1','caption_2',...,'caption_n']
  
  ## embed caption by CLIP
  def embedCaps(self, caps):
    device = self.args.device
    texts = []
    for cap in caps:
      texts.append(cap)
    texts = clip.tokenize(texts).to(device)
    return texts
  def clip_word_emb(self,text,clip_model):
    x = clip_model.token_embedding(text).type(clip_model.dtype)  # [batch_size, n_ctx, d_model]
    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.ln_final(x).type(clip_model.dtype)
    # 求出sent len
    sent_len = min(30,max(text.argmax(dim=-1)))
    # print('sent_len:',sent_len)
    # print('text_atgmax:',text.argmax(dim=-1))
    # 新的word embedding
    y = x[:,:sent_len,:]
    #print('y.shape:',y.shape)
    #p = clip_model.text_projection
    #print('p.shape:',p.shape)
    return y.permute(0,2,1)
  texts = self.embedCaps(caps)
  word_embs = self.clip_word_emb(texts,self.clip_model).detach()
  
  ## prepare the mask input
  def get_mask(self,mask_path,device):
    def get_mul_mask(mask):
      mask_size = [16,64,256]
      mask_transform1 = transforms.Compose([
        transforms.Resize(mask_size[0]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
      ])
      mask_transform2 = transforms.Compose([
        transforms.Resize(mask_size[1]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
      ])
      mask_transform3 = transforms.Compose([
        transforms.Resize(mask_size[2]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
      ])
      mask1 = mask_transform1(mask).unsqueeze(0).to(device)
      mask2 = mask_transform2(mask).unsqueeze(0).to(device)
      mask3 = mask_transform3(mask).unsqueeze(0).to(device)
      return [mask1,mask2,mask3]
    img = Image.open(mask_path).convert('RGB')
    width, height = img.size
    mask_list = get_mul_mask(img)
    return mask_list
  masks = self.get_mask(mask_path,device)
  
  ## inference
  ori_sample, _, _ = g_ema(masks,word_embs,noise=ori_noise)
  
  ## save img
  ## plase use your own saveName
  self.save_grid_images(
  					ori_sample, 
  					saveName
  				)
  
  
  
  

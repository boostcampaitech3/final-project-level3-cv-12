

def main():

    while True:

        new_img = from_camera(interval > inference_time)
        is_human = human_detect(img)
        
        if first:
            before_img = new_img
            before_lamen2bbox = now_lamen2bbox
            continue         
        
        if is_human:
            continue
        
        else:
            bboxes = detect(img) # [bbox1, bbox2....]
            now_lamen2bbox = classification(img, bboxes) # {라면1:[(bbox1, state), (bbox2, state)], 라면2:[(bbox1, state), (bbox2, state)]...}
            
            lamen2samebbox= discriminator(
                before_img, before_labmen2bbox, 
                new_img, new_lamen2bbox
                ) # IoU계산 #{라면1: [(before_category_idx, now_category_idx, state_score)... ]}
            
            # out of stock detection 
            brgt_output = check_brightness(lamen2samebbox)
            depth_output = check_depth()

            ensemble_output = ensemble(outputs)
            '''
            [-1, 0] -> -0.5 -> alert?
            '''

            if out_of_stock:
                notification()


            before_img = new_img
            before_lamen2bbox = now_lamen2bbox
addpath(genpath('gifti-master'))
addpath(genpath('cifti-matlab-master'))

%%%%%%%%%%  read in the Schaefer atlas
atlas_L='Atlas/Schaefer2018_1000Parcels_7Networks_order.L.dlabel.nii'
atlas_R='Atlas/Schaefer2018_1000Parcels_7Networks_order.R.dlabel.nii'
LShaefer_parcellation = ft_read_cifti(atlas_L);
RShaefer_parcellation = ft_read_cifti(atlas_R);
LShaefer_parcellation = LShaefer_parcellation.parcels'; LShaefer_parcellation(isnan(LShaefer_parcellation)) = 0;
RShaefer_parcellation = RShaefer_parcellation.parcels'; RShaefer_parcellation(isnan(RShaefer_parcellation)) = 0;
Shaefer_parcellation = [ LShaefer_parcellation RShaefer_parcellation ];
Shaefer_parcellation_temp = Shaefer_parcellation*0; 
count = 1;
    for i = 1 : max(Shaefer_parcellation)
        
        if(sum(Shaefer_parcellation==i))
            Shaefer_parcellation_temp(Shaefer_parcellation==i) = count;
            count = count + 1;
        else
            i
        end
    end
    Shaefer_parcellation = Shaefer_parcellation_temp;


 %%%%%%%%%%%%  read in atlas and reconstruct the data
 maskl = gifti('Atlas/L.atlasroi.10k_fs_LR.shape.gii'); maskl = maskl.cdata;
 maskr = gifti('Atlas/R.atlasroi.10k_fs_LR.shape.gii'); maskr = maskr.cdata;
mask = logical([ maskl' maskr' ]);

% start reading in the RS data and reconstruct.
datafolder='**************************************/data_HCP_gradient_676_rsfMRI';
filelist=dir([datafolder,'/*/*.nii']);
for i = 1:length(filelist)
    file=[filelist(i).folder,'/',filelist(i).name]
    if exist(strrep(file,'.nii','_TS_Parcel.txt'),'file')==2
        continue
    end
    try
    %file='/100206/rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean_flt.10k.dtseries.nii'
        cii=ciftiopen(file);
        ts_temp1 = cii.cdata';
        selected_shape_idx = find(mask);
        ts_1 = zeros(size(ts_temp1, 1), size(mask, 2));
        ts_1(:, selected_shape_idx) = ts_temp1(:, 1:length(selected_shape_idx));
        ts_1 = ts_1(11:end, :);

        % extract time series from Schaeffer atlas
        unique_parcel=unique(Shaefer_parcellation);
        num_pacel=length(unique_parcel);
        ts_parcel=zeros(size(ts_1,1),(num_pacel-1));
        for i = 2:num_pacel
            ts_parcel(:,i-1)=mean(ts_1(:,Shaefer_parcellation==unique_parcel(i)),2)';
        end
        %save the time*parcel oupt.
        dlmwrite(strrep(file,'.nii','_TS_Parcel.txt'),ts_parcel)
    catch
         a=0
    end
end


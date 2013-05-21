
    %%%%%%%%%%%%%%%%%%%% show intialization
    for i=1:size(dataWeightAll,1)
      [~, ind]=max(dataWeightAll(i,:));
      clusters(ind).imageIndex=[clusters{ind}.imageIndex,i];
    end
    
    %%%%% generating html
    fid = fopen(['result0.html'], 'wt'); 

    for c=1:numCluster
       for i=1:size(cluster{c},2)
           id=cluster{c}.imageIndex(i);
           fprintf(fid, '%s\n', ['<IMG SRC="' fullfile(inPath, files(id).name) '" height=70 ' 'width=70>']); 
       end
       fprintf(fid, '%s\n', ['<br>']);
       
       fprintf(fid, '%s\n', ['<hr>']);
    end

    fprintf(fid, '%s\n', ['<br>']);
    fprintf(fid, '%s\n', ['<hr>']);
    fclose(fid);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
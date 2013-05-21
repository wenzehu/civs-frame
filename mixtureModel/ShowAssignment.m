    %%%%% generating html
    fid = fopen(fullfile(resultPath, ['result' num2str(it) '.html']), 'wt');
    for c=1:numCluster
       for i=1:size(clusters(c).imageIndex,2)
           id=clusters(c).imageIndex(i);
           fprintf(fid, '%s\n', ['<IMG SRC="' fullfile('./img', files(id).name) '" height=70 ' 'width=70>']); 
       end
       fprintf(fid, '%s\n', ['<br>']);
       
       fprintf(fid, '%s\n', ['<hr>']);
    end

    fprintf(fid, '%s\n', ['<br>']);
    fprintf(fid, '%s\n', ['<hr>']);
    fclose(fid);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

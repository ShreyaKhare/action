function [ A ] = read_feature( cur_file )
fwid=fopen(cur_file);
fseek(fwid,0,'bof');
sam_num=fread(fwid,1,'float');
dim=fread(fwid,1,'float');

A=zeros(dim,sam_num);
for i=1:sam_num
	    A(:,i)=fread(fwid,dim,'float');
    end

    fseek(fwid,(4+sam_num*dim*4),'bof');
    a=fread(fwid,1,'float');
    last=A(dim,sam_num);

    fclose(fwid); 
end
